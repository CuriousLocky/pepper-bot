import logging

from pepperbot.telegram.app import PepperBotApplication


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    PepperBotApplication().run()


if __name__ == "__main__":
    main()
