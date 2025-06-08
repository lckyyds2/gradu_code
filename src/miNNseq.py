
# 导入必要的库
import numpy as np  # 导入numpy用于数值计算
from sklearn.metrics.pairwise import nan_euclidean_distances  # 导入用于计算带有缺失值的欧氏距离的函数
from tqdm import tqdm  # 导入进度条显示库


def miNNseq(X: np.array, n_neighbors: int, squared: bool = True, lamda: float = 0.3):
    """
    使用基于最近邻的方法进行序列数据的缺失值插补
    
    参数:
    X: 输入的二维numpy数组，包含缺失值(np.nan)
    n_neighbors: 用于插补的最近邻数量
    squared: 是否使用平方欧氏距离
    lamda: 高斯核函数的带宽参数
    
    返回:
    完成插补后的数组
    """
    X = X.copy()  # 创建输入数据的副本，避免修改原始数据
    imputed_indices = np.where(np.isnan(X))  # 获取所有缺失值的索引位置

    # 计算样本间的欧氏距离矩阵
    distances = nan_euclidean_distances(
        X, X, squared=squared, missing_values=np.nan)
    
    # 对每个缺失值进行插补
    for i, j in tqdm(zip(imputed_indices[0], imputed_indices[1])):
        # 获取距离最近的样本索引
        neighbors_idx = np.argsort(distances[i])
        # 排除同一位置上有缺失值的样本
        neighbors_idx = neighbors_idx[~np.isin(
            neighbors_idx, imputed_indices[0][np.where(imputed_indices[1] == j)[0]])]
        neighbors_idx = neighbors_idx[:n_neighbors]  # 选择前n_neighbors个最近邻
        
        # 计算基于距离的权重（使用高斯核函数）
        weights = (1/np.sqrt(2*np.pi))*np.exp((-1/2) *
                                              np.power(distances[i][neighbors_idx] / lamda, 2))
        # 处理所有权重为0的特殊情况
        weights = weights + 1 if np.all(weights == 0) else weights
        normalized_weights = weights / np.sum(weights)  # 归一化权重

        # 使用加权平均进行插补
        X[i, j] = np.average(X[neighbors_idx, j], weights=normalized_weights)
        # 更新距离矩阵
        distances[i, :] = distances[:, i] = nan_euclidean_distances(
            X[i][np.newaxis, :], X, squared=squared, missing_values=np.nan)[0]

    return X


if __name__ == "__main__":
    # 创建示例数据
    data = np.array([[1, 2, np.nan], [3, np.nan, 3], [
                    7, 6, 5], [np.nan, 8, 7], [2, np.nan, 4]])

    # 使用miNNseq进行缺失值插补
    imputed_data = miNNseq(data, n_neighbors=3)
    print("Imputed Data:\n", imputed_data)
