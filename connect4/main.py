import argparse
from c4.gui_pygame import GameGUI


def main():
    parser = argparse.ArgumentParser(description='Connect 4 with AI (Pygame)')
    parser.add_argument('--ai', choices=['minimax', 'qlearn'], default='minimax', help='AI type for the computer')
    parser.add_argument('--depth', type=int, default=5, help='Depth for minimax AI')
    parser.add_argument('--q-path', type=str, default=None, help='Path to load a trained Q-learning agent (.pkl)')
    parser.add_argument('--human-first', action='store_true', help='Make human play first (default True)')
    parser.add_argument('--computer-first', action='store_true', help='Make computer play first')
    args = parser.parse_args()

    human_first = True
    if args.computer_first:
        human_first = False
    elif args.human_first:
        human_first = True

    gui = GameGUI(ai_type=args.ai, depth=args.depth, q_path=args.q_path, human_first=human_first)
    gui.run()


if __name__ == '__main__':
    main()

