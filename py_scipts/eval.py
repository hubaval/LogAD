import argparse
from pathlib import Path
from datasets import Dataset, disable_progress_bars, load_from_disk
from py_scipts.utils import cut_timestamp, CustomModel, EnumEnv, check, EnumDatasource, pd, np, get_enum_timestamp_value, batch_process

def main():
    directory_path = Path(args_pars.out)
    checkpoint = any(directory_path.iterdir())
    predict = True

    CModel.load()

    # Verify checkpoint
    if checkpoint:
        data = load_from_disk(args_pars.out)
        print(f"Found existing checkpoint {args_pars.out}")
        df = data.to_pandas()

        if 'full_predicted' in df.columns:
            predict = df['full_predicted'].isnull().any()

    else:
        data = load_from_disk(args_pars.dataset)[args_pars.split]
        df = data.to_pandas()

        # Cap dataset size if needed
        if args_pars.cap_size > 0:
            df = df.iloc[:args_pars.cap_size].reset_index(drop=True)

        print(f"Starting predictions from {args_pars.dataset}")

    # Verify if prediction is needed or just evaluation
    if predict:
        input_chunks, targets_chunks = zip(*batch_process(df['input_text'].to_numpy(), df['target_text'].to_numpy(), args_pars.chunk))
        input_chunks = list(input_chunks)

        pred_list = []
        iteration_start = 0

        # If checkpoint prediction is not complete, then resume predictions from last
        if checkpoint and 'checkpoint' in df.columns:
            pred_list = df.loc[df['full_predicted'].notna(), 'full_predicted'].tolist()
            iteration_start = int(df['checkpoint'].unique()[0]) + 1
            print(f"Resuming checkpoint predictions from {args_pars.dataset} at iteration {iteration_start}")

        # Prediction iterations
        for i in range(iteration_start, len(input_chunks)):
            predicted = CModel.predict(input_chunks[i], temperature=args_pars.temperature, max_new_tokens=args_pars.max_new_tokens)
            pred_list += predicted

            if i % 50 == 0:
                df['full_predicted'] = pred_list + [pd.NA] * (len(df) - len(pred_list))
                df['checkpoint'] = i

                checkpoint_dt = Dataset.from_pandas(df)
                print(f"Saving checkpoint predictions in {args_pars.out} at iteration {i}, length : {len(pred_list)}/{df.shape[0]}")
                checkpoint_dt.save_to_disk(f"{args_pars.out}")

        df.drop('checkpoint', axis=1, inplace=True)
        df['full_predicted'] = pred_list
        df['full_target_text'] = df['target_text']

        # Saving predictions
        dt = Dataset.from_pandas(df)
        print(f"Saving checkpoint predictions in {args_pars.out}")
        dt.save_to_disk(f"{args_pars.out}")
    else:
        print("Found existing prediction in dataset")

    # Verify if logs need to be cut (removing timestamp, extra information ..)
    if args_pars.cut_size is not None:
        df[['target_text', 'targ_tmstp']] = df.apply(lambda row: pd.Series(
            cut_timestamp(
                row['full_target_text'],
                n_split=int(args_pars.cut_size) if args_pars.cut_size.isdigit()
                else get_enum_timestamp_value(row['source'])  # Use the function here
            )), axis=1)
        df[["predicted", "pred_tmstp"]] = df.apply(lambda row: pd.Series(
            cut_timestamp(
                row['full_predicted'],
                n_split=int(args_pars.cut_size) if args_pars.cut_size.isdigit()
                else get_enum_timestamp_value(row['source'])
            )), axis=1)

        no_bracketlist = [EnumDatasource.OPENSTACK.value]
        df.loc[df['source'].isin(no_bracketlist), 'predicted'] = df.loc[df['source'].isin(no_bracketlist), 'predicted'].str.replace(r'\[.*?\]', '', regex=True)
        df.loc[df['source'].isin(no_bracketlist), 'target_text'] = df.loc[df['source'].isin(no_bracketlist), 'target_text'].str.replace(r'\[.*?\]', '', regex=True)

    # Compute cosine similarity scores for evaluation
    df['similarity'] = df.apply(
        lambda row: CModel.cosine_similarity_numpy(row['target_text' if args_pars.cut_size else 'full_target_text'], row['predicted' if args_pars.cut_size else 'full_predicted']), axis=1)
    df['similarity'] = df['similarity'].astype(np.float32)

    total_match = df['similarity'].sum()
    total_samples = len(df)

    check(total_match, total_samples)

    # Saving predictions and results
    dt = Dataset.from_pandas(df)
    print(f"Saving in {args_pars.out}")
    dt.save_to_disk(f"{args_pars.out}")


def setup_parser():
    """
    Parsing script parameters
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store", type=str, help="dataset path to tokenize")
    parser.add_argument("--out", action="store", type=str, help="Output")
    parser.add_argument("--split", action="store", type=str, help="split to use", default="train, test, verif")
    parser.add_argument("--model", action="store", type=str, help="Path to the model to evaluate", default="hubaval/llama-3.1-8B-fttlogs")
    parser.add_argument("--cut_size", action="store", help="which part to cut (timestamp), n_split or auto")
    parser.add_argument("--max_new_tokens", action="store", type=int, help="max number of new tokens - default (128)")
    parser.add_argument("--temperature", action="store", type=float, help="Specify prediction temperature (default 0.2)", default=0.2)
    parser.add_argument("--chunk", action="store", type=int, help="Specify evaluation chunk size (default 3)", default=3)
    parser.add_argument("--quantize", action="store", type=str, help="Quantize (awq  or bnb) or not", default=None)
    parser.add_argument("--cap_size", action="store", type=int, help="Give evaluation max capping size on used split", default=0)
    return parser


if __name__ == "__main__":
    print("\nStarting evaluation script....")
    args_pars = setup_parser().parse_args()
    disable_progress_bars()

    # Get model with custom parameters (Quantization, ...)
    CModel = CustomModel(args_pars.model, device="auto", environment=EnumEnv.EVALUATION, quantize=args_pars.quantize)

    main()
