#!/usr/bin/env python3
"""
Interactive command builder for generate_cache.py

This CLI wizard guides you through building a generate_cache.py command
with all the right flags based on what you want to accomplish.
"""

import subprocess
import sys
import os

# Try to import questionary for beautiful prompts
try:
    import questionary
    from questionary import Style
    USE_QUESTIONARY = True

    # Custom style
    custom_style = Style([
        ('qmark', 'fg:#673ab7 bold'),
        ('question', 'bold'),
        ('answer', 'fg:#f44336 bold'),
        ('pointer', 'fg:#673ab7 bold'),
        ('highlighted', 'fg:#673ab7 bold'),
        ('selected', 'fg:#cc5454'),
        ('separator', 'fg:#cc5454'),
        ('instruction', ''),
        ('text', ''),
    ])
except ImportError:
    USE_QUESTIONARY = False
    print("⚠️  Note: Install 'questionary' for a better experience: pip install questionary\n")


def show_banner():
    """Display welcome banner"""
    banner = """
╔════════════════════════════════════════════════════════════╗
║     NFL Data Pipeline - Interactive Command Builder       ║
╚════════════════════════════════════════════════════════════╝

This wizard will help you build a generate_cache.py command
by asking a series of questions about what you want to do.

Press Ctrl+C at any time to exit.
"""
    print(banner)


def ask_yes_no(question, default=True):
    """Ask a yes/no question"""
    if USE_QUESTIONARY:
        return questionary.confirm(question, default=default, style=custom_style).ask()
    else:
        suffix = "[Y/n]" if default else "[y/N]"
        response = input(f"{question} {suffix}: ").strip().lower()
        if not response:
            return default
        return response in ('y', 'yes')


def ask_choice(question, choices, default=None):
    """Ask a multiple choice question"""
    if USE_QUESTIONARY:
        return questionary.select(question, choices=choices, default=default, style=custom_style).ask()
    else:
        print(f"\n{question}")
        for i, choice in enumerate(choices, 1):
            marker = ">" if choice == default else " "
            print(f"  {marker} {i}. {choice}")

        while True:
            response = input(f"Select (1-{len(choices)}): ").strip()
            if not response and default:
                return default
            try:
                idx = int(response) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            except ValueError:
                pass
            print("Invalid selection, try again.")


def ask_text(question, default=""):
    """Ask a text input question"""
    if USE_QUESTIONARY:
        return questionary.text(question, default=default, style=custom_style).ask()
    else:
        prompt = f"{question}"
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        response = input(prompt).strip()
        return response if response else default


def ask_questions():
    """Ask all questions and return command options dict"""
    options = {}

    print("\n📊 DATA FILES")
    print("─" * 60)
    options['skip_data'] = not ask_yes_no("Update data files (schedule, stats, standings)?", default=True)

    print("\n🎲 SIMULATIONS")
    print("─" * 60)
    run_sims = ask_yes_no("Run simulations?", default=True)
    options['skip_sims'] = not run_sims

    if run_sims:
        sim_choice = ask_choice(
            "How many simulations?",
            choices=["Quick (100)", "Standard (1,000)", "Full (10,000)", "Custom"],
            default="Standard (1,000)"
        )

        if "Quick" in sim_choice:
            options['simulations'] = 100
        elif "Standard" in sim_choice:
            options['simulations'] = 1000
        elif "Full" in sim_choice:
            options['simulations'] = 10000
        else:
            while True:
                custom = ask_text("Enter number of simulations", default="1000")
                try:
                    options['simulations'] = int(custom)
                    if options['simulations'] > 0:
                        break
                    print("⚠️  Please enter a positive number")
                except ValueError:
                    print("⚠️  Please enter a valid number")

        use_seed = ask_yes_no("Use random seed for reproducibility?", default=False)
        if use_seed:
            while True:
                seed = ask_text("Enter seed value", default="42")
                try:
                    options['seed'] = int(seed)
                    if options['seed'] >= 0:
                        break
                    print("⚠️  Please enter a non-negative number")
                except ValueError:
                    print("⚠️  Please enter a valid number")

    print("\n🤖 AI ANALYSIS - TEAMS")
    print("─" * 60)
    gen_team_ai = ask_yes_no("Generate team AI analysis?", default=True)
    options['skip_team_ai'] = not gen_team_ai

    if gen_team_ai:
        team_choice = ask_choice(
            "Which teams?",
            choices=["All teams", "Specific teams (comma-separated)", "Regenerate all teams", "Regenerate specific teams"],
            default="All teams"
        )

        if "Regenerate all" in team_choice:
            options['regenerate_team_ai'] = "all"
        elif "Regenerate specific" in team_choice:
            teams = ask_text("Enter team abbreviations (comma-separated, e.g., DET,MIN,GB)", default="")
            if teams:
                options['regenerate_team_ai'] = teams
        elif "Specific teams" in team_choice:
            teams = ask_text("Enter team abbreviations (comma-separated, e.g., DET,MIN,GB)", default="")
            if teams:
                options['regenerate_team_ai'] = teams

    print("\n🤖 AI ANALYSIS - GAMES")
    print("─" * 60)
    gen_game_ai = ask_yes_no("Generate game AI analysis?", default=True)
    options['skip_game_ai'] = not gen_game_ai

    if gen_game_ai:
        game_choice = ask_choice(
            "Which games?",
            choices=["All games", "Analysis only (completed)", "Preview only (upcoming)", "Specific ESPN IDs", "Regenerate all", "Regenerate analysis", "Regenerate preview"],
            default="All games"
        )

        if "Regenerate all" in game_choice:
            options['regenerate_game_ai'] = "all"
        elif "Regenerate analysis" in game_choice:
            options['regenerate_game_ai'] = "analysis"
        elif "Regenerate preview" in game_choice:
            options['regenerate_game_ai'] = "preview"
        elif "Specific ESPN IDs" in game_choice:
            espn_ids = ask_text("Enter ESPN IDs (comma-separated, e.g., 401772856,401772855)", default="")
            if espn_ids:
                options['regenerate_game_ai'] = espn_ids
        elif "Analysis only" in game_choice:
            options['regenerate_game_ai'] = "analysis"
        elif "Preview only" in game_choice:
            options['regenerate_game_ai'] = "preview"

    print("\n🚀 DEPLOYMENT")
    print("─" * 60)
    options['commit'] = ask_yes_no("Commit changes to git?", default=False)
    options['deploy_netlify'] = ask_yes_no("Deploy to Netlify?", default=False)
    options['deploy_render'] = ask_yes_no("Deploy to Render?", default=False)

    print("\n⚙️  ADVANCED")
    print("─" * 60)
    options['test_mode'] = ask_yes_no("Run in test mode (disable AI calls)?", default=False)

    return options


def build_command(options):
    """Build generate_cache.py command from options dict"""
    cmd_parts = ["python", "generate_cache.py"]

    # Data options
    if options.get('skip_data'):
        cmd_parts.append("--skip-data")

    # Simulation options
    if options.get('skip_sims'):
        cmd_parts.append("--skip-sims")
    else:
        if 'simulations' in options:
            cmd_parts.append(f"--simulations {options['simulations']}")
        if 'seed' in options:
            cmd_parts.append(f"--seed {options['seed']}")

    # Team AI options
    if options.get('skip_team_ai'):
        cmd_parts.append("--skip-team-ai")
    elif 'regenerate_team_ai' in options:
        cmd_parts.append(f'--regenerate-team-ai "{options["regenerate_team_ai"]}"')

    # Game AI options
    if options.get('skip_game_ai'):
        cmd_parts.append("--skip-game-ai")
    elif 'regenerate_game_ai' in options:
        cmd_parts.append(f'--regenerate-game-ai "{options["regenerate_game_ai"]}"')

    # Deployment options
    if options.get('commit'):
        cmd_parts.append("--commit")
    if options.get('deploy_netlify'):
        cmd_parts.append("--deploy-netlify")
    if options.get('deploy_render'):
        cmd_parts.append("--deploy-render")

    # Advanced options
    if options.get('test_mode'):
        cmd_parts.append("--test-mode")

    return " ".join(cmd_parts)


def display_command(cmd):
    """Display command in a nice formatted box"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║  Command to run:" + " " * 60 + "║")
    print("║" + " " * 78 + "║")

    # Word wrap the command if it's too long
    max_width = 74
    words = cmd.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_length + word_len > max_width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_len
        else:
            current_line.append(word)
            current_length += word_len

    if current_line:
        lines.append(" ".join(current_line))

    for line in lines:
        padding = max_width - len(line)
        print("║  " + line + " " * padding + "  ║")

    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()


def run_command(cmd):
    """Execute the command with subprocess"""
    print("\n🚀 Running command...\n")
    print("─" * 60)

    try:
        # Run the command and show output in real-time
        result = subprocess.run(cmd, shell=True, check=False)

        print("\n" + "─" * 60)
        if result.returncode == 0:
            print("✅ Command completed successfully!")
        else:
            print(f"⚠️  Command exited with code {result.returncode}")

        return result.returncode == 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Command interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Error running command: {e}")
        return False


def main():
    """Main entry point"""
    try:
        show_banner()
        options = ask_questions()
        cmd = build_command(options)
        display_command(cmd)

        # Ask if they want to run it
        run_now = ask_yes_no("Run this command now?", default=True)

        if run_now:
            success = run_command(cmd)
            if success:
                print("\n✓ All done!")
            else:
                print("\n⚠️  Command did not complete successfully")
                sys.exit(1)
        else:
            print("\n✓ Command ready to copy!")
            print("  You can run it manually by copying the command above.")

    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
