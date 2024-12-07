import os
import cv2
import time
import json
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ultralytics import YOLO
from multiprocessing import Process, Queue
from tenacity import retry, stop_after_attempt, wait_exponential


def load_s3_jsonl(s3_obj_list_file):
    s3_files_list = []
    with open(s3_obj_list_file, "r") as f:
        for line in f:
            json_object = json.loads(line.rstrip("\n"))
            s3_path, url = list(json_object.items())[0]
            s3_files_list.append((s3_path, url))

    return s3_files_list

def filter_unprocessed_files(s3_obj_list_file, parquet_file_path):
    s3_files_list = load_s3_jsonl(s3_obj_list_file)

    if not os.path.exists(parquet_file_path):
        return s3_files_list
    else:
        table = pq.read_table(parquet_file_path)
        processed_files = table.column("file_path").to_pylist()
        print(f'processed_files: {len(processed_files) / len(s3_files_list)}')
        unprocessed_files = [(file_path, url) for file_path, url in s3_files_list if file_path not in processed_files]
        return unprocessed_files
    
def get_data_type(data_list):
    """
    根据数据列表推断合适的Arrow数据类型。
    """
    if all(isinstance(x, bytes) for x in data_list):
        return pa.binary()
    return pa.string() if data_list else pa.null()

def update_existing_table(existing_table, new_table):
    """
    根据file_path列来更新已存在表中的同名行数据。
    """
    # 将 Arrow 表转换为 pandas DataFrame，方便进行数据操作（此处可根据实际性能需求考虑是否有更优转换方式）
    existing_df = existing_table.to_pandas()
    new_df = new_table.to_pandas()

    # 遍历新表中的每一行数据（根据file_path列来判断和更新）
    for index, row in new_df.iterrows():
        file_path_value = row['file_path']
        existing_index = existing_df[existing_df['file_path'] == file_path_value].index
        if len(existing_index) > 0:
            # 如果存在同名行，更新对应行的数据
            existing_index = existing_index[0]
            for col in existing_df.columns:
                if col!= 'file_path' and row[col] != None:
                    existing_df.at[existing_index, col] = row[col]
        else:
            # 如果不存在同名行，则添加该行到现有表对应的数据结构中（这里添加到pandas DataFrame）
            existing_df = pd.concat([existing_df, row.to_frame().T], ignore_index=True)

    # 将更新后的pandas DataFrame转换回Arrow表
    updated_table = pa.Table.from_pandas(existing_df)
    return updated_table

def write_lists_to_parquet(lists_data, file_path):
    """
    将包含不同元素的列表数据写入Parquet文件。

    参数:
    lists_data (list of lists): 包含多个列表的列表，每个子列表代表一次生成的数据
    file_path (str): Parquet文件的存储路径
    """
    field_names = ["file_path", "det_label", "det_bbox", "det_conf"]
    all_data = [[] for _ in range(len(field_names))]

    # 遍历每个生成的列表，将元素填充到对应的字段数据列表中
    for single_list in lists_data:
        filled_data = [None] * len(field_names)
        for index, element in enumerate(single_list):
            if index < len(field_names):
                if isinstance(element, list):
                    filled_data[index] = json.dumps(element)
                elif isinstance(element, bytes):
                    filled_data[index] = element
                else:
                    filled_data[index] = element

        for index, value in enumerate(filled_data):
            all_data[index].append(value)

    # 根据实际收集到的数据情况构建 Arrow 数组
    arrays = []
    for index, data_list in enumerate(all_data):
        data_type = get_data_type(data_list)
        arrays.append(pa.array(data_list, type=data_type))

    # 构建模式（Schema）
    fields = []
    for name, data_list in zip(field_names, all_data):
        field_type = get_data_type(data_list)
        fields.append(pa.field(name, field_type))
    schema = pa.schema(fields)

    # 构建 Arrow 表
    table = pa.Table.from_arrays(arrays, schema=schema)

    # 写入 Parquet 文件逻辑优化及异常处理添加
    try:
        if os.path.exists(file_path):
            # 如果文件已存在，尝试以追加模式写入
            existing_table = pq.read_table(file_path)
            if "file_path" in existing_table.column_names:
                # 处理同名 file_path 行更新逻辑
                new_table = update_existing_table(existing_table, table)
                pq.write_table(new_table, file_path, use_dictionary=True, compression='snappy')
            else:
                with pq.ParquetWriter(file_path, schema, use_dictionary=True, compression='snappy', append=True) as writer:
                    writer.write_table(table)
        else:
            # 如果文件不存在，直接写入新文件
            pq.write_table(table, file_path, use_dictionary=True, compression='snappy')
    except FileNotFoundError as e:
        print(f"文件不存在导致写入Parquet文件出错: {e}")
        raise
    except pq.ParquetException as e:
        print(f"Parquet文件格式相关错误: {e}")
        raise
    except Exception as e:
        print(f"写入Parquet文件时出现其他未知错误: {e}")
        raise

def dumb_fill_in(unprocessed_files_list, parquet_path):
    unprocessed_files = [item[0] for item in unprocessed_files_list]
    if len(unprocessed_files) > 0:
        if os.path.exists(parquet_path):
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
        else:
            columns = ["file_path", "det_label", "det_bbox", "det_conf"]
            
            df = pd.DataFrame(columns=columns)

        new_data = pd.DataFrame({
        "file_path": unprocessed_files,
        "det_label": [None] * len(unprocessed_files),
        "det_bbox": [None] * len(unprocessed_files),
        "det_conf": [None] * len(unprocessed_files),
    })
        updated_df = pd.concat([df, new_data], ignore_index=True)

                # 将更新后的DataFrame转换为Arrow Table
        updated_table = pa.Table.from_pandas(updated_df)

        update_path = parquet_path.split(".parquet")[0] + "_update.parquet"

        pq.write_table(updated_table, update_path)
        print("填充完成")
    else:
        print("处理完成，无剩余")

@retry(stop=stop_after_attempt(5), reraise=True, wait=wait_exponential(multiplier=1, min=1, max=4))
def sample_frames(video_path, frame_sample_ratio):
    """
    Extract frames from video at a given interval.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_sample_ratio == 0:
            yield frame_idx, frame
        frame_idx += 1

    cap.release()

def file_reading_worker(file_list, frame_sample_ratio, batch_size, read_queue):
    """
    单个文件读取worker，负责从视频中提取帧并组成批次放入队列
    """
    for tuple in file_list:
        video_path = tuple[1]
        video_name = os.path.basename(tuple[0]).split('.')[0]
        batch_data = []
        batch_metadata = []

        for frame_idx, frame in sample_frames(video_path, frame_sample_ratio):
            batch_data.append(frame)
            batch_metadata.append((video_name, video_path, frame_idx))

            if len(batch_data) == batch_size:
                read_queue.put((batch_data, batch_metadata))
                batch_data = []
                batch_metadata = []

        if batch_data:
            read_queue.put((batch_data, batch_metadata))

    # read_queue.put(None)  # 确保放入结束信号    

def file_reading_stage(file_list, frame_sample_ratio, batch_size, read_queues):
    """
    文件读取阶段，分配任务到多个文件读取worker
    """
    num_workers = len(read_queues)
    chunk_size = len(file_list) // num_workers
    processes = []
    try:
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_workers - 1 else len(file_list)
            worker_file_list = file_list[start_idx:end_idx]
            read_queue = read_queues[i]
            p = Process(target=file_reading_worker, args=(worker_file_list, frame_sample_ratio, batch_size, read_queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 向所有读取队列放入结束信号
        for read_queue in read_queues:
            read_queue.put(None)
    except Exception as e:
        print(f"文件读取阶段出现异常: {e}")

def model_inference_worker(model_path, read_queue, inference_queue):
    """
    单个模型推理worker，从读取队列获取批次数据进行推理并将结果放入推理队列
    """
    try:
        model = YOLO(model_path)
        while True:
            item = read_queue.get()
            if item is None:  # 结束信号
                print('Read queue is dead')
                inference_queue.put(None)
                break
            batch_data, batch_metadata = item
            results = model.predict(batch_data, device='cuda', save=False, batch=len(batch_data))
            inference_queue.put((results, batch_metadata))
    except Exception as e:
        print(f"模型推理worker出现异常: {e}")

def model_inference_stage(model_path, read_queues, inference_queue):
    """
    模型推理阶段，分配任务到多个模型推理worker
    """
    num_workers = len(read_queues)
    processes = []
    try:
        for i in range(num_workers):
            read_queue = read_queues[i]
            p = Process(target=model_inference_worker, args=(model_path, read_queue, inference_queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 向推理队列放入结束信号
        inference_queue.put(None)
    except Exception as e:
        print(f"模型推理阶段出现异常: {e}")

def result_saving_stage(s3_obj_list_file, inference_queue, output_path):
    """
    Result saving stage: Aggregate results and save to Parquet.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_results = {}
    try:
        while True:
            item = inference_queue.get()
            if item is None:  # End signal
                print('inferer is dead')
                break

            results, metadata = item
            for i, result in enumerate(results):
                video_name, video_path, frame_idx = metadata[i]

                if video_name not in video_results:
                    video_results[video_name] = {
                        "file_path": video_name,
                        "det_label": [],
                        "det_bbox": [],
                        "det_conf": [],
                    }

                # Collect detection details
                frame_labels = []
                frame_bboxes = []
                frame_confs = []

                for box in result.boxes:
                    frame_labels.append(int(box.cls.cpu().numpy()))
                    frame_bboxes.append(box.xyxy.cpu().numpy().tolist()[0])
                    frame_confs.append(float(box.conf.cpu().numpy()))

                video_results[video_name]["det_label"].append(frame_labels)
                video_results[video_name]["det_bbox"].append(frame_bboxes)
                video_results[video_name]["det_conf"].append(frame_confs)

        # Save results to Parquet
        data = [
            [
                video_name,
                results["det_label"],
                results["det_bbox"],
                results["det_conf"]
            ]
            for video_name, results in video_results.items()
        ]
    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        print("Saving existing results before exiting...")
        print(f'The progress Stop at {video_name}')
        write_lists_to_parquet(data, output_path)
        file_list = filter_unprocessed_files(s3_obj_list_file, output_path)
        dumb_fill_in(file_list, output_path)
    finally:
        write_lists_to_parquet(data, output_path)
        file_list = filter_unprocessed_files(s3_obj_list_file, output_path)
        dumb_fill_in(file_list, output_path)
        print('finished!')

def process_file_list(s3_obj_list_file, file_list, output_path, model_path, frame_sample_ratio, batch_size, num_worker):
    """
    Main process: Coordinate all stages using multiprocessing with multiple workers.
    """
    # 创建多个读取队列和一个推理队列
    read_queues = [Queue(maxsize=10000) for _ in range(num_worker)]
    inference_queue = Queue(maxsize=10000)

    # 启动文件读取阶段进程
    reader = Process(target=file_reading_stage,
                     args=(file_list, frame_sample_ratio, batch_size, read_queues))
    reader.start()

    # 启动模型推理阶段进程
    inferencer = Process(target=model_inference_stage,
                         args=(model_path, read_queues, inference_queue))
    inferencer.start()

    # 启动结果保存阶段进程
    saver = Process(target=result_saving_stage, args=(s3_obj_list_file, inference_queue, output_path))
    saver.start()

    # 等待所有进程完成
    reader.join()
    inferencer.join()
    saver.join()
    
    # 显式关闭队列，释放资源
    for queue in read_queues:
        queue.close()
    inference_queue.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and save detection results.")
    parser.add_argument("--s3_obj_list_file", required=True, help="Path to the directory containing videos.")
    parser.add_argument("--target_dir", required=True, help="Path to the output Parquet file.")
    parser.add_argument("--model_path", default="/root/model/yolo/yolov10x.pt", help="Path to YOLO model.")
    parser.add_argument("--frame_sample_ratio", type=int, default=30, help="Frame sampling ratio.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for model inference.")
    parser.add_argument("--num_worker", type=int, default=8, help="Number of workers.")
    args = parser.parse_args()

    s3_files_list = filter_unprocessed_files(args.s3_obj_list_file, args.target_dir)
    start = time.time()
    process_file_list(args.s3_obj_list_file, s3_files_list, args.target_dir, args.model_path, args.frame_sample_ratio, args.batch_size,
                      args.num_worker)
    end = time.time()
    print(end-start, 's')
