# -*- coding: utf-8 -*-
"""
数据预处理与邻接构建脚本（中文注释）
功能：
 1) 读取 /mnt/data 下的 milano/trentino CSV（第一列为时间戳，后面为节点列）
 2) 填充缺失值、标准化并保存为 numpy (T, N) 以及 meta.json（保存均值与标准差）
 3) 读取对应的 geojson（若存在），尝试匹配节点名并基于经纬度构建 geo_adj（kNN + inverse distance）
 4) 也计算基于 Pearson 相关性的 corr_adj（top-k），并生成 combined_adj = alpha*geo + (1-alpha)*corr
 输出：
  - /mnt/data/preprocessed_for_st/<city>_norm.npy
  - /mnt/data/preprocessed_for_st/<city>_nodes_meta.json
  - /mnt/data/preprocessed_for_st/<city>_geo_adj.npy
  - /mnt/data/preprocessed_for_st/<city>_corr_adj.npy
  - /mnt/data/preprocessed_for_st/<city>_combined_adj.npy
"""
import os
import numpy as np
import pandas as pd
import json
from difflib import get_close_matches
from math import radians, sin, cos, asin, sqrt

# 可配置参数
DATA_DIR = "../data"
OUT_DIR = os.path.join(DATA_DIR, "preprocessed_for_st")
os.makedirs(OUT_DIR, exist_ok=True)
CITIES = [
    ("milano_traffic_nid.csv", "milano.geojson", "milano"),
    ("trentino_traffic_nid.csv", "trento.geojson", "trentino"),
]
K = 8          # kNN 的 k
ALPHA = 0.6    # geo 与 corr 的融合权重（0~1），越接近 1 越重地理信息
FUZZY_CUT = 0.68  # 模糊匹配阈值

# Haversine 函数（向量化版本）
def haversine_km(lat1, lon1, lat2, lon2):
    # lat/lon in degrees, returns km
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# 读取 CSV，第一列为时间
def load_csv(path):
    df = pd.read_csv(path)
    # 第一列假定为时间索引
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    # 强制数值
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# 填充 + 标准化
def fill_and_norm(df):
    # 时间序列插值填充
    df = df.interpolate(method='time').ffill().bfill()
    mu = df.mean()
    sigma = df.std().replace(0, 1.0)
    df_norm = (df - mu) / sigma
    return df_norm, mu.to_dict(), sigma.to_dict()

# 计算基于相关性的 top-k 邻接（绝对值）
def build_corr_adj(df_norm, k=8):
    corr = df_norm.corr().fillna(0.0).values  # N x N
    N = corr.shape[0]
    corr_abs = np.abs(corr)
    adj = np.zeros_like(corr_abs)
    for i in range(N):
        idx = np.argsort(-corr_abs[i])
        chosen = [j for j in idx if j != i][:k]
        adj[i, chosen] = corr_abs[i, chosen]
    adj = np.maximum(adj, adj.T)
    if adj.max() > 0:
        adj = adj / adj.max()
    return adj

# 简单模糊匹配节点名（CSV 列名 -> geojson properties）
def fuzzy_match(node_list, geo_names, cutoff=0.68):
    mapping = {}
    for node in node_list:
        if node in geo_names:
            mapping[node] = geo_names.index(node)
        else:
            matches = get_close_matches(node, geo_names, n=1, cutoff=cutoff)
            if matches:
                mapping[node] = geo_names.index(matches[0])
            else:
                mapping[node] = None
    return mapping

# 主流程：单城市处理
def process_city(csv_fname, geo_fname, city_label):
    csv_path = os.path.join(DATA_DIR, csv_fname)
    geo_path = os.path.join(DATA_DIR, geo_fname)
    if not os.path.exists(csv_path):
        print(f"[跳过] 未找到 CSV: {csv_path}")
        return
    print(f"处理城市: {city_label}")
    df = load_csv(csv_path)
    df_norm, mu, sigma = fill_and_norm(df)
    # 保存标准化后的 csv/npy
    df_norm.to_csv(os.path.join(OUT_DIR, f"{city_label}_norm.csv"))
    np.save(os.path.join(OUT_DIR, f"{city_label}_norm.npy"), df_norm.values)  # T x N
    # 保存 meta
    meta = {"mu": mu, "sigma": sigma, "nodes": list(df_norm.columns), "shape": df_norm.shape}
    with open(os.path.join(OUT_DIR, f"{city_label}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=lambda x: x if not hasattr(x, "to_dict") else x.to_dict())
    # 计算相关性 adjacency
    corr_adj = build_corr_adj(df_norm, k=K)
    np.save(os.path.join(OUT_DIR, f"{city_label}_corr_adj.npy"), corr_adj)
    # 若 geojson 存在，尝试读取并匹配
    geo_adj = None
    try:
        import geopandas as gpd
        if os.path.exists(geo_path):
            gdf = gpd.read_file(geo_path)
            # 标准化 CRS
            if gdf.crs is None:
                try:
                    gdf = gdf.set_crs("EPSG:4326")
                except Exception:
                    pass
            else:
                try:
                    gdf = gdf.to_crs("EPSG:4326")
                except Exception:
                    pass
            # 提取可作为名字的字段
            geo_names = []
            for idx, row in gdf.iterrows():
                nm = None
                for key in ['name', 'NAME', 'NOME', 'nome', 'id', 'ID', 'Name']:
                    if key in row and pd.notnull(row[key]):
                        nm = str(row[key]); break
                if nm is None:
                    nm = f"__idx_{idx}"
                geo_names.append(nm)
            # 模糊匹配
            nodes = list(df_norm.columns)
            mapping = fuzzy_match(nodes, geo_names, cutoff=FUZZY_CUT)
            # 收集匹配结果并构建坐标数组
            lats, lons, matched = [], [], []
            for node in nodes:
                idx = mapping.get(node)
                if idx is None:
                    lats.append(np.nan); lons.append(np.nan); matched.append(False)
                else:
                    geom = gdf.geometry.iloc[idx]
                    centroid = geom.centroid if geom.geom_type != 'Point' else geom
                    lons.append(float(centroid.x)); lats.append(float(centroid.y)); matched.append(True)
            matched = np.array(matched)
            matched_count = int(matched.sum())
            print(f"Geo 匹配：{matched_count}/{len(nodes)} 节点匹配成功")
            # 如果匹配点足够（至少 3），构建 geo_adj
            if matched_count >= 3:
                valid_idx = np.where(matched)[0]
                lon_valid = np.array(lons)[valid_idx].astype(float)
                lat_valid = np.array(lats)[valid_idx].astype(float)
                # 计算距离矩阵
                N = len(nodes)
                geo_adj = np.zeros((N, N), dtype=float)
                # 计算 valid 距离矩阵
                VV = len(valid_idx)
                # 先构建 VV x VV 距离矩阵
                lat1 = np.repeat(lat_valid[:, None], VV, axis=1)
                lon1 = np.repeat(lon_valid[:, None], VV, axis=1)
                lat2 = lat1.T; lon2 = lon1.T
                dists = haversine_km(lat1, lon1, lat2, lon2)  # km
                np.fill_diagonal(dists, np.inf)
                for i_i, i in enumerate(valid_idx):
                    idxs = np.argsort(dists[i_i])[:K]
                    chosen = valid_idx[idxs]
                    weights = 1.0 / (dists[i_i, idxs] + 1e-6)
                    geo_adj[i, chosen] = weights
                geo_adj = np.maximum(geo_adj, geo_adj.T)
                if geo_adj.max() > 0:
                    geo_adj = geo_adj / geo_adj.max()
                np.save(os.path.join(OUT_DIR, f"{city_label}_geo_adj.npy"), geo_adj)
            else:
                print("匹配点太少，跳过 geo_adj 构建（使用 corr_adj）")
        else:
            print(f"未找到 GeoJSON: {geo_path}，仅使用相关性邻接")
    except Exception as e:
        print("未安装 geopandas 或读取 geojson 失败，跳过 geo 处理。错误：", e)
    # 生成 combined（若 geo_adj 存在）
    if geo_adj is not None:
        combined = ALPHA * geo_adj + (1.0 - ALPHA) * corr_adj
    else:
        combined = corr_adj
    np.save(os.path.join(OUT_DIR, f"{city_label}_combined_adj.npy"), combined)
    # 保存 nodes meta（匹配详情）
    nodes_meta = []
    for i, node in enumerate(list(df_norm.columns)):
        nodes_meta.append({
            "node": node,
            "matched": bool((geo_adj is not None) and (not np.isnan(geo_adj[i]).all())),
            "lon": None if geo_adj is None or np.isnan(geo_adj[i]).all() else None
        })
    with open(os.path.join(OUT_DIR, f"{city_label}_nodes_meta.json"), "w") as f:
        json.dump(nodes_meta, f, indent=2)
    print(f"{city_label} 处理完成，结果保存在 {OUT_DIR}")

if __name__ == "__main__":
    for csv_fname, geo_fname, label in CITIES:
        process_city(csv_fname, geo_fname, label)
    print("全部处理结束。")
