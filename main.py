import argparse
from config import load_config, apply_overrides
from training import train_model
from plotting import plot_emiss_vs_temp, predict_cumulative_temperature


def main() -> None:
    parser = argparse.ArgumentParser(description="CESM emulator entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Run model training")
    train_p.add_argument("--config", type=str, required=True, help="Path to config file")
    train_p.add_argument("--set", nargs="*", default=[], help="Override config values")

    plot_p = sub.add_parser("plot", help="Generate plots")
    plot_p.add_argument("--mode", choices=["emiss_vs_temp", "cumulative"], required=True, help="Plot type")

    args = parser.parse_args()

    if args.command == "train":
        cfg = load_config(args.config)
        apply_overrides(cfg, args.set)
        train_model(cfg)
    elif args.command == "plot":
        if args.mode == "emiss_vs_temp":
            plot_emiss_vs_temp()
        else:
            predict_cumulative_temperature(
                ckpt_path="runs/exp3/checkpoints/ckpt_epoch_0070.pt",
                cond_file="../CESM2-LESN_emulator/co2_final.nc",
                cond_var="CO2_em_anthro",
                target_file="../CESM2-LESN_emulator/splits/fold_1/climate_data_train_fold1.nc",
                target_var="TREFHT",
            )


if __name__ == "__main__":
    main()
