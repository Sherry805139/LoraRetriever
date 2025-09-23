from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from InstructorEmbedding import INSTRUCTOR

# Global variables to store the FAISS index and the embedding model
global_index = None
global_model = None
model_names = None

instruction = "Represent the sentence for similar task retrieval: "

def initialize_index(models, model_size='7b'):
    """
    Initialize the FAISS index for model retrieval. This function:
    - Loads a global embedding model.
    - Computes average embeddings for each model based on provided samples.
    - Creates and populates a FAISS index with the model embeddings.

    Parameters:
    - models: A list of dictionaries, each containing a 'model_name' and 'sample' entries.
              'sample' should be a list of dictionaries with 'inputs'.
    - model_size: A string indicating the model size (e.g., '7b', '13b') for naming conventions.
    """
    global global_index, global_model, model_names

    # Load the embedding model for retrieval
    global_model = INSTRUCTOR('Styxxxx/lora_retriever', device='cpu')

    all_model_embeddings = []
    model_names = []

    # Compute average embeddings for each model
    for model in models:
        if model_size == '7b':
            model_name = f"Styxxxx/llama2_7b_lora-{model['model_name']}"
        else:
            model_name = f"Styxxxx/llama2_13b_lora-{model['model_name']}"

        model_names.append(model_name)
        model_samples = []

        # Collect sample inputs for each model
        for sample in model['sample']:
            sample_context = sample['inputs']
            model_samples.append([instruction, sample_context])

        # Compute embeddings for the model's samples and take the mean
        embeddings = get_embeddings(model_samples)
        average_embedding = np.mean(embeddings, axis=0)
        all_model_embeddings.append(average_embedding)

    # model_name里是所有models的模型名
    # model_samples里是所有models的instruction和input
    # all_model_embeddings里是所有models的embedding

    # Create a FAISS index with the collected embeddings
    # 把所有embeddings堆叠到一个矩阵中
    all_model_embeddings = np.vstack(all_model_embeddings)
    # d是每个embeddings的维度
    d = all_model_embeddings.shape[1]
    # 创建一个基于内积的平坦索引（内积越大 = 向量越相似）
    global_index = faiss.IndexFlatIP(d)
    # global_index是检索数据库，包含所有模型的向量
    global_index.add(all_model_embeddings)

def get_embeddings(text_list):
    """
    Encode a list of text samples using the global embedding model.

    Parameters:
    - text_list: A list of texts to be encoded. Each element should be [instruction, text].
    """
    return global_model.encode(text_list)

def perform_search(query_list, k=20, exclude_list=None):
    """
    Perform a similarity search for each query in the provided query_list using the global FAISS index.
    Returns a list of retrieved model names and a binary mapping matrix indicating which models
    were retrieved for each query.

    Parameters:
    - query_list: List of query strings to retrieve similar tasks/models for.
    - k: The number of top results to retrieve for each query.
    - exclude_list: An optional list of models to exclude for each query.

    Returns:
    - all_results_list: A list of all retrieved unique model names across all queries.
    - mapping_matrix: A binary matrix (list of lists) indicating which models were retrieved for each query.
    """
    global global_index, model_names

    all_results_set = set() # 空集合，用于去重
    query_to_results_map = {} # 空字典，存储查询到结果的映射

    # Perform search for each query
    for j, query in enumerate(query_list):
        # 1. 生成查询文本的嵌入向量
        query_embedding = get_embeddings([[instruction, query]])[0] # 取第一个（也是唯一一个）向量
        # instruction应为全局/传入的指令前缀，比如“检索相似模型：”
        
        # 2. 用FAISS的库函数search，检索k+1个结果，返回distances（相似度分数）和indices（索引位置）
        distances, indices = global_index.search(np.array([query_embedding]), k+1)

        # 3. 初始化标记和当前查询的结果列表
        contained = False # 标记：当前查询是否需要排除某个模型
        results = []

        # 4. 遍历搜索到的k+1个结果，处理排除逻辑
        for i, idx in enumerate(indices[0]):
            # indices[0]：当前查询的所有结果索引（因为只查了1个查询，所以取第0维）
            
            # 特殊情况：如果遍历到第k个结果（即原本的Topk），但还没处理排除，说明排除的模型不在前k个，直接跳过
            if i == k and not contained:
                # If we reached the last allowed result but haven't accounted for exclusion yet, skip
                continue

            model_name = model_names[idx]

            # Exclude specific model for this query if applicable
            # 如果有exclude_list，且当前模型是该查询要排除的，跳过并标记contained
            if exclude_list and model_name == exclude_list[j]:
                contained = True
                continue
            
            # 把符合条件的模型名加入全局去重集合和当前查询结果列表
            all_results_set.add(model_name)
            results.append(model_name)

        query_to_results_map[query] = results

    # Convert all results to a list and construct a mapping matrix
    # 1. 集合转列表：得到所有查询的去重模型名列表（集合自动去重，列表方便后续按顺序索引）
    all_results_list = list(all_results_set)

    # 2. 构建二进制匹配矩阵（行=查询，列=去重模型）
    mapping_matrix = []

    for query in query_list:
        # 对于所有all_results_list中的（已经去重）模型，如果在query查询得到的模型集合中，标记为1
        mapping_vector = [1 if result in query_to_results_map[query] 
                            else 0 
                            for result in all_results_list]
        mapping_matrix.append(mapping_vector)

    return all_results_list, mapping_matrix