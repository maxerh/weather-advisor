import argparse

from src.data_factory.feature_engineering import FeatureGenerator
from src.data_factory.dataset_windowed import create_datasets_from_splits
from torch.utils.data import DataLoader


def get_dataloader(path_to_dataset, batch_size=32, input_length=168, prediction_length=24,
                   target_cols=None):
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
        target_cols=target_cols,
    )

    # 5. Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, fg, df_train, df_val, df_test


def get_trainer(dl_train, dl_val, dl_test, fg, df_train, df_val, input_length, prediction_length, epochs, args):
    # For deep models:
    if args.model == 'lstm':
        from src.trainers.lstm_trainer import LSTMTrainer
        trainer = LSTMTrainer(
            input_dim=dl_train.dataset.data.shape[1],
            output_dim=len(dl_train.dataset.target_cols),
            pred_len=prediction_length,
            hidden_dim=64,
            num_layers=2,
            train_loader=dl_train,
            val_loader=dl_val,
            test_loader=dl_test,
            epochs=epochs,
        )

    elif args.model == 'transformer':
        from src.trainers.transformer_trainer import TransformerTrainer
        trainer = TransformerTrainer(
            input_dim=dl_train.dataset.data.shape[1],
            output_dim=len(dl_train.dataset.target_cols),
            pred_len=prediction_length,
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            num_layers=2,
            train_loader=dl_train,
            val_loader=dl_val,
            test_loader=dl_test,
            epochs=epochs,
        )

    elif args.model == 'deeptransformer':
        from src.trainers.deep_transformer_trainer import TransformerForecastTrainer
        trainer = TransformerForecastTrainer(
            input_dim=dl_train.dataset.data.shape[1],
            output_dim=len(dl_train.dataset.target_cols),
            input_length=input_length,
            pred_len=prediction_length,
            embed_dim=128,
            num_heads=8,
            ff_dim=256,
            num_layers=4,
            dropout=0.1,
            train_loader=dl_train,
            val_loader=dl_val,
            test_loader=dl_test,
            epochs=epochs,
            learning_rate=5e-5,
        )

    elif args.model == 'autoformer':
        from src.trainers.autoformer_trainer import AutoformerTrainer
        trainer = AutoformerTrainer(
            input_dim=dl_train.dataset.data.shape[1],
            output_dim=len(dl_train.dataset.target_cols),
            input_length=input_length,
            pred_length=prediction_length,
            d_model=128,
            n_encoder_layers=3,
            n_decoder_layers=2,
            top_k=5,
            ff_dim=256,
            dropout=0.1,
            train_loader=dl_train,
            val_loader=dl_val,
            test_loader=dl_test,
            epochs=epochs,
            learning_rate=5e-5,
        )

    elif args.model == 'prophet':
        # Prophet does *not* use PyTorch DataLoader
        from src.trainers.prophet_trainer import ProphetTrainer
        # We extract the ds (time) and y (target) from raw dataframe
        # Use the FeatureGenerator data (before normalization) or df_train
        # Assuming target_cols has exactly one column
        target = args.target_cols[0]
        ds_series = fg.dataframe["time"]
        y_series = fg.dataframe[target]

        # Split according to train/val proportions
        # Use the same indices from df_train / df_val
        # We'll use df_train and df_val passed in
        ds_train = df_train["time"]
        y_train = df_train[target]
        ds_val = df_val["time"]
        y_val = df_val[target]

        trainer = ProphetTrainer(
            ds=ds_train,
            y=y_train,
            freq="H",  # hourly data
            prophet_kwargs=dict(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
            )
        )

        # We'll also need val data for evaluation
        # We'll wrap it into the trainer or handle separately

    else:
        raise ValueError("Invalid model selected. Choose from [lstm, transformer, deeptransformer, autoformer, prophet].")

    return trainer


def get_evaluator(dl_val, model_name):
    if model_name == 'lstm':
        from src.evaluators.lstm_evaluator import LSTMEvaluator
        evaluator = LSTMEvaluator(
            model_name=model_name,
            dataloader=dl_val,
        )
    elif model_name == 'transformer':
        from src.evaluators.transformer_evaluator import TransformerEvaluator
        evaluator = TransformerEvaluator(
            model_name=model_name,
            dataloader=dl_val,
        )
    elif model_name == 'deeptransformer':
        from src.evaluators.deep_transformer_evaluator import DeepTransformerEvaluator
        evaluator = DeepTransformerEvaluator(
            model_name=model_name,
            dataloader=dl_val,
        )
    elif model_name == 'autoformer':
        from src.evaluators.autoformer_evaluator import AutoformerEvaluator
        evaluator = AutoformerEvaluator(
            model_name=model_name,
            dataloader=dl_val,
        )
    elif model_name == 'prophet':
        # TODO pass trainer and val dataframe or load trainer in evaluator?
        from src.evaluators.prophet_evaluator import ProphetEvaluator
        #evaluator = ProphetEvaluator(
        #    model_name=model_name,
        #    ds_val=df_val["time"],
        #    y_val=df_val[target_cols[0]]
        #)
    else:
        raise ValueError("Invalid model selected. Choose from [lstm, transformer, deeptransformer].")
    return evaluator


def main(args):
    input_length = 168      # past 7 days
    prediction_length = 24  # next day
    epochs = 25

    # Pass target_cols from args
    target_cols = ['temperature_2m']

    dl_train, dl_val, dl_test, fg, df_train, df_val, df_test = get_dataloader(
        "data/history/munich_weather_2015_2024.csv",
        batch_size=32,
        input_length=input_length,
        prediction_length=prediction_length,
        target_cols=target_cols
    )

    if args.mode == 'train':
        print("Running in training mode...")

        trainer = get_trainer(
            dl_train, dl_val, dl_test,
            fg, df_train, df_val,
            input_length, prediction_length, epochs, args
        )

        if args.model == 'prophet':
            # Prophet train + evaluate
            trainer.train()
            forecast = trainer.predict(periods=len(df_val))
            metrics = trainer.evaluate(val_ds=df_val["time"], val_y=df_val[target_cols[0]])
            print("Prophet validation metrics:", metrics)
        else:
            trainer.train()
    elif args.mode == 'eval':
        print("Running in evaluation mode...")
        evaluator = get_evaluator(dl_val, args.model)
        evaluator.evaluate()
    elif args.mode == 'test':
        print("Running in test mode...")
        # Add testing logic here
    else:
        print("Invalid mode selected. Choose from [train, eval, test].")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        help='Train, evaluation or test mode [train,eval,test]', default='eval')
    parser.add_argument('--model', type=str,
                        help='Which model to use [lstm, transformer, deeptransformer, autoformer]', default='autoformer')

    args = parser.parse_args()
    main(args)