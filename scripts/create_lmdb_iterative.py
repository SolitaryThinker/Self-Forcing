from tqdm import tqdm
import numpy as np
import argparse
import torch
import lmdb
import glob
import os

# from utils.lmdb import store_arrays_to_lmdb, process_data_dict
def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        image_shape = txn.get(f"{array_name}_shape".encode()).decode()
        image_shape = tuple(map(int, image_shape.split()))
    return image_shape


def store_arrays_to_lmdb(env, arrays_dict, start_index=0):
    """
    Store rows of multiple numpy arrays in a single LMDB.
    Each row is stored separately with a naming convention.
    """
    with env.begin(write=True) as txn:
        for array_name, array in arrays_dict.items():
            for i, row in enumerate(array):
                # Convert row to bytes
                if isinstance(row, str):
                    row_bytes = row.encode()
                else:
                    row_bytes = row.tobytes()

                data_key = f'{array_name}_{start_index + i}_data'.encode()

                txn.put(data_key, row_bytes)


def process_data_dict_with_text_embedding(data_dict, seen_prompts, indexing_type=None):
    output_dict = {}

    all_videos = []
    all_prompts = []
    all_text_embeddings = []

    prompt = data_dict['prompts']
    if prompt in seen_prompts:
        return
    else:
        seen_prompts.add(prompt)

    video = data_dict['ode_latent']
    text_embedding = data_dict['text_embedding']

    if indexing_type is not None:
        if indexing_type == 'high':
            video = video[:, [0, 6, 12, 18, -1]]
        elif indexing_type == 'low':
            video = video[:, [26, 29, 32, 35, -1]]
        else:
            raise ValueError(f"Invalid indexing type: {indexing_type}")
    else:
        assert video.shape[1] == 5

    video = video.half().numpy()
    text_embedding = text_embedding.cpu().numpy()
    all_videos.append(video)
    all_prompts.append(prompt)
    all_text_embeddings.append(text_embedding)
    all_videos = np.concatenate(all_videos, axis=0)
    all_prompts = np.array(all_prompts)
    all_text_embeddings = np.array(all_text_embeddings)

    output_dict['latents'] = all_videos
    output_dict['prompts'] = all_prompts
    output_dict['text_embeddings'] = all_text_embeddings

    return output_dict


def process_data_dict(data_dict, seen_prompts, indexing_type=None):
    output_dict = {}

    all_videos = []
    all_prompts = []
    # print('data_dict', data_dict.keys())
    if 'prompts' in data_dict.keys() and 'ode_latent' in data_dict.keys():
        # print('data_dict', data_dict['prompts'], data_dict['ode_latent'].shape)
        assert data_dict['ode_latent'].shape[1] == 5
        data_dict = {data_dict['prompts']: data_dict['ode_latent']}

    for prompt, video in data_dict.items():
        if prompt in seen_prompts:
            continue
        else:
            seen_prompts.add(prompt)

        if indexing_type is not None:
            if indexing_type == 'high':
                video = video[:, [0, 6, 12, 18, -1]]
            elif indexing_type == 'low':
                video = video[:, [26, 29, 32, 35, -1]]
            else:
                raise ValueError(f"Invalid indexing type: {indexing_type}")
        # video = video[:, [0, 6, 12, 18, -1]]

        # print('video', video.shape)
        # video = video[:, [0, 6, 12, 18, -1]]
        # video = video[:, [26, 29, 32, 35, -1]]
        # print('video', video.shape)
        video = video.half().numpy()
        all_videos.append(video)
        all_prompts.append(prompt)

    if len(all_videos) == 0:
        return {"latents": np.array([]), "prompts": np.array([])}

    all_videos = np.concatenate(all_videos, axis=0)

    output_dict['latents'] = all_videos
    output_dict['prompts'] = np.array(all_prompts)

    return output_dict


def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, row_index, shape=None):
    """
    Retrieve a specific row from a specific array in the LMDB.
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    if dtype == str:
        array = row_bytes.decode()
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)

    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array



def main():
    """
    Aggregate all ode pairs inside a folder into a lmdb dataset.
    Each pt file should contain a (key, value) pair representing a
    video's ODE trajectories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        required=True, help="path to ode pairs")
    parser.add_argument("--lmdb_path", type=str,
                        required=True, help="path to lmdb")
    parser.add_argument("--indexing_type", type=str,
                        required=False, choices=['high', 'low', None], help="indexing type", default=None)
    parser.add_argument("--include_text_embedding",
                        action='store_true',
                        required=False, help="whether to include text embedding", default=False)
    args = parser.parse_args()
    print(f"args: {args}")

    all_files_file = os.path.join(args.data_path, 'all_files.txt')
    if not os.path.exists(all_files_file):
        print(f"all_files_file does not exist, generating {all_files_file}")
        # all_files = sorted(glob.glob(os.path.join(args.data_path, "*.pt"), recursive=True))
        all_files = sorted(glob.glob('**/*.pt', root_dir=args.data_path, recursive=True))
        # print(f"all_files: {all_files}")
        all_files = [os.path.join(args.data_path, file) for file in all_files]
        with open(all_files_file, 'w') as f:
            for file in all_files:
                f.write(file + '\n')
    else:
        print(f"all_files_file exists, reading from {all_files_file}")
        with open(all_files_file, 'r') as f:
            all_files = [line.strip() for line in f]

    print('len(all_files)', len(all_files))

    # figure out the maximum map size needed
    total_array_size = 5000000000000  # adapt to your need, set to 5TB by default

    counter = 0
    env = lmdb.open(args.lmdb_path, map_size=total_array_size * 2)

    if args.include_text_embedding:
        # latents, prompts, text_embeddings
        num_metadata = 3
    else:
        # latents, prompts
        num_metadata = 2

    with env.begin() as txn:
        myList = [ key for key, _ in txn.cursor() ]
        print(myList)
        print(len(myList))
        assert len(myList) % num_metadata == 0
        if 'latents_shape' in myList and 'prompts_shape' in myList:
            print("latents_shape and prompts_shape exist")
            counter = (len(myList)-num_metadata) // num_metadata
        else:
            print("latents_shape and prompts_shape do not exist")
            counter = len(myList) // num_metadata


    seen_prompts = set()  # for deduplication

    print(f"skipping the first {counter} files")
    # skip the first counter files
    all_files = all_files[counter:]

    for index, file in tqdm(enumerate(all_files)):
        # read from disk
        # try:
        data_dict = torch.load(file)

        if args.include_text_embedding:
            data_dict = process_data_dict_with_text_embedding(data_dict, seen_prompts, indexing_type=args.indexing_type)
        else:
            data_dict = process_data_dict(data_dict, seen_prompts, indexing_type=args.indexing_type)

        # write to lmdb file
        store_arrays_to_lmdb(env, data_dict, start_index=counter)
        counter += len(data_dict['prompts'])
        # except Exception as e:
        #     print(f"Error processing {file}: {e}")
        #     continue
        # if index > 1400: 
        # if index > 1: 
        #     break
    # print('len(data_dict)', len(data_dict))
    # save each entry's shape to lmdb
    with env.begin(write=True) as txn:
        for key, val in data_dict.items():
            # print(key, val)
            array_shape = np.array(val.shape)
            array_shape[0] = counter

            shape_key = f"{key}_shape".encode()
            shape_str = " ".join(map(str, array_shape))
            txn.put(shape_key, shape_str.encode())


if __name__ == "__main__":
    main()
