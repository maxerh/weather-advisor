import argparse

from src.data_factory.feature_engineering import FeatureGenerator
from src.data_factory.dataset_windowed import create_datasets_from_splits
from torch.utils.data import DataLoader


def get_dataaloader(path_to_dataset, batch_size=32, input_length=168, prediction_length=24):
    # 1. Create features
    fg = FeatureGenerator(path_to_dataset)
    fg.create_features()

    # 2. Split
    df_train, df_val, df_test = fg.train_val_test_split()

    # 3. Normalize (fit on train only!)
    fg.fit_normalization(df_train)
    df_train = fg.transform(df_train)
    df_val = fg.transform(df_val)
    df_test = fg.transform(df_test)

    # 4. Create datasets
    train_ds, val_ds, test_ds = create_datasets_from_splits(
        df_train, df_val, df_test,
        input_length=input_length,             # past 7 days
        prediction_length=prediction_length,   # next 24 hours
    )

    # 5. Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def main(args):

    input_length = 168      # past 7 days
    prediction_length = 24  # next day
    epochs = 10
    dl_train, dl_val, dl_test = get_dataaloader("data/history/munich_weather_2015_2024.csv",
                                                batch_size=32,
                                                input_length=input_length,
                                                prediction_length=prediction_length,
                                                )

    if args.model == 'lstm':
        from src.trainers.lstm_trainer import LSTMTrainer
        trainer = LSTMTrainer(
            input_dim=dl_train.dataset.data.shape[1],
            output_dim=len(dl_train.dataset.target_cols),
            pred_len=24,
            hidden_dim=64,
            num_layers=2,
            train_dataloader=dl_train,
            val_dataloader=dl_val,
            test_dataloader=dl_test,
            epochs=epochs,
        )
    elif args.model == 'transformer':
        from src.trainers.transformer_trainer import TransformerTrainer
        trainer = TransformerTrainer(
            input_dim=dl_train.dataset.data.shape[1],
            output_dim=len(dl_train.dataset.target_cols),
            pred_len=24,
            embed_dim=64,
            num_heads=4,
            hidden_dim=128,
            num_layers=2,
            train_dataloader=dl_train,
            val_dataloader=dl_val,
            test_dataloader=dl_test,
            epochs=epochs,
        )
    else:
        raise ValueError("Invalid model selected. Choose from [lstm, transformer].")

    if args.mode == 'train':
        print("Running in training mode...")
        # Add training logic here
        trainer.train()
    elif args.mode == 'eval':
        print("Running in evaluation mode...")
        # Add evaluation logic here
    elif args.mode == 'test':
        print("Running in test mode...")
        # Add testing logic here
    else:
        print("Invalid mode selected. Choose from [train, eval, test].")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        help='Train, evaluation or test mode [train,eval,test]', default='train')
    parser.add_argument('--model', type=str,
                        help='Which model to use [lstm, transformer]', default='transformer')

    args = parser.parse_args()
    main(args)