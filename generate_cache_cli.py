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
║     NFL Data Pipeline - Interactive Command Builder        ║
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

    print("\n🎯 MODE SELECTION")
    print("─" * 60)

    mode_choice = ask_choice(
        "What do you want to do?",
        choices=[
            "Generate new files (full pipeline)",
            "Deploy existing files only (skip all generation)"
        ],
        default="Generate new files (full pipeline)"
    )

    if "Deploy existing" in mode_choice:
        # Deploy-only mode - skip all generation
        options['deploy_only'] = True
        # Skip to deployment section
        print("\n🚀 DEPLOYMENT")
        print("─" * 60)
        options['commit'] = ask_yes_no("Commit changes to git?", default=True)
        options['deploy_netlify'] = ask_yes_no("Deploy to Netlify?", default=True)

        copy_files = ask_yes_no("Copy data files to local directory?", default=True)
        if copy_files:
            copy_path = ask_text("Copy destination path", default="/home/obliojoe/source/whoshouldlose2/public/data")
            if copy_path:
                options['copy_to'] = copy_path

        return options

    # Continue with normal generation flow
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
            choices=["Test (1,000)", "Standard (10,000)", "High (50,000)", "Very High (100,000)", "Custom"],
            default="Standard (10,000)"
        )

        if "Test" in sim_choice:
            options['simulations'] = 1000
        elif "Standard" in sim_choice:
            options['simulations'] = 10000
        elif "High" in sim_choice:
            options['simulations'] = 50000
        elif "Very High" in sim_choice:
            options['simulations'] = 100000
        else:
            while True:
                custom = ask_text("Enter number of simulations", default="10000")
                try:
                    options['simulations'] = int(custom)
                    if options['simulations'] > 0:
                        break
                    print("⚠️  Please enter a positive number")
                except ValueError:
                    print("⚠️  Please enter a valid number")

        seed = ask_text("Enter simulation seed [random]", default="")
        if seed:
            while True:
                try:
                    options['seed'] = int(seed)
                    if options['seed'] >= 0:
                        break
                    print("⚠️  Please enter a non-negative number")
                    seed = ask_text("Enter simulation seed [random]", default="")
                    if not seed:
                        break
                except ValueError:
                    print("⚠️  Please enter a valid number")
                    seed = ask_text("Enter simulation seed [random]", default="")
                    if not seed:
                        break

    print("\n🤖 AI ANALYSIS - TEAMS")
    print("─" * 60)
    print("(Team AI should be regenerated whenever simulations or data changes)")

    team_choice = ask_choice(
        "Regenerate team AI analysis?",
        choices=["All teams (recommended)", "Specific teams only", "Skip team AI"],
        default="All teams (recommended)"
    )

    if "Skip" in team_choice:
        options['skip_team_ai'] = True
    elif "Specific" in team_choice:
        teams = ask_text("Enter team abbreviations (comma-separated, e.g., DET,MIN,GB)", default="")
        if teams:
            options['regenerate_team_ai'] = teams
        else:
            # If they don't enter any teams, regenerate all
            options['regenerate_team_ai'] = "all"
    else:  # All teams
        options['regenerate_team_ai'] = "all"

    print("\n🤖 AI ANALYSIS - GAMES")
    print("─" * 60)
    print("(Default: generate missing + convert previews to analysis when games complete)")

    game_choice = ask_choice(
        "Game AI options:",
        choices=[
            "Generate new/updated (missing + preview→analysis)",
            "Skip game AI (use existing)",
            "Regenerate all previews (upcoming games)",
            "Regenerate all analysis (completed games)",
            "Regenerate everything (all games)",
            "Regenerate for specific team(s)",
            "Regenerate specific ESPN IDs"
        ],
        default="Generate new/updated (missing + preview→analysis)"
    )

    if "Skip" in game_choice:
        options['skip_game_ai'] = True
    elif "new/updated" in game_choice:
        # Generate missing games + convert preview→analysis when games complete
        # This is the default behavior when neither skip nor regenerate is specified
        pass
    elif "all previews" in game_choice:
        options['regenerate_game_ai'] = "preview"
    elif "all analysis" in game_choice:
        options['regenerate_game_ai'] = "analysis"
    elif "everything" in game_choice:
        options['regenerate_game_ai'] = "all"
    elif "specific team" in game_choice:
        teams = ask_text("Enter team abbreviations (comma-separated, e.g., DET,MIN,GB)", default="")
        if teams:
            options['regenerate_game_ai'] = f"team:{teams}"
    elif "specific ESPN IDs" in game_choice:
        espn_ids = ask_text("Enter ESPN IDs (comma-separated, e.g., 401772856,401772855)", default="")
        if espn_ids:
            options['regenerate_game_ai'] = espn_ids

    print("\n🤖 AI ANALYSIS - DASHBOARD")
    print("─" * 60)

    dashboard_choice = ask_choice(
        "Generate dashboard content?",
        choices=["Yes (generate dashboard)", "Skip dashboard"],
        default="Yes (generate dashboard)"
    )

    if "Skip" in dashboard_choice:
        options['skip_dashboard_ai'] = True

    # AI Model selection (only if ANY AI generation is enabled)
    needs_model = (
        not options.get('skip_team_ai') or
        not options.get('skip_game_ai') or
        not options.get('skip_dashboard_ai')
    )

    if needs_model:
        print("\n🧠 AI MODEL SELECTION")
        print("─" * 60)

        model_choice = ask_choice(
            "Which AI model to use?",
            choices=[
                "Default (haiku - fast, cheap)",
                "Opus (claude-opus-4-1 - highest quality)",
                "Sonnet (claude-sonnet-4-5 - balanced)",
                "Sonnet 3.7 (claude-3-7-sonnet-latest)",
                "GPT-5 (gpt-5 - highest quality)",
                "GPT-5 Mini (gpt-5-mini - balanced)",
                "GPT-4o (gpt-4o)",
                "GPT-4o Mini (gpt-4o-mini)",
                "Custom model name"
            ],
            default="Default (haiku - fast, cheap)"
        )

        if "Opus" in model_choice:
            options['ai_model'] = "opus"
        elif model_choice.startswith("Sonnet 3.7"):
            options['ai_model'] = "sonnet-3.7"
        elif "Sonnet" in model_choice:
            options['ai_model'] = "sonnet"
        elif model_choice.startswith("GPT-5 Mini"):
            options['ai_model'] = "gpt-5-mini"
        elif model_choice.startswith("GPT-5 ("):
            options['ai_model'] = "gpt-5"
        elif model_choice.startswith("GPT-4o Mini"):
            options['ai_model'] = "gpt-4o-mini"
        elif "GPT-4o" in model_choice:
            options['ai_model'] = "gpt-4o"
        elif "Custom" in model_choice:
            custom_model = ask_text("Enter model alias or full name", default="")
            if custom_model:
                options['ai_model'] = custom_model

    print("\n🚀 DEPLOYMENT")
    print("─" * 60)
    options['commit'] = ask_yes_no("Commit changes to git?", default=False)
    options['deploy_netlify'] = ask_yes_no("Deploy to Netlify?", default=False)

    copy_files = ask_yes_no("Copy data files to local directory?", default=True)
    if copy_files:
        copy_path = ask_text("Copy destination path", default="/home/obliojoe/source/whoshouldlose2/public/data")
        if copy_path:
            options['copy_to'] = copy_path

    print("\n⚙️  ADVANCED")
    print("─" * 60)

    # Worker configuration
    configure_workers = ask_yes_no("Configure worker counts for parallel processing?", default=True)
    if configure_workers:
        print("\nWorker Configuration:")
        print("  Team AI workers: Number of parallel threads for team analysis")
        print("  Game AI workers: Number of parallel threads for game analysis")
        print("  Higher = faster but uses more API tokens/minute")

        while True:
            team_workers = ask_text("Team AI workers [32]", default="32")
            try:
                options['team_ai_workers'] = int(team_workers)
                if options['team_ai_workers'] > 0:
                    break
                print("⚠️  Please enter a positive number")
            except ValueError:
                print("⚠️  Please enter a valid number")

        while True:
            game_workers = ask_text("Game AI workers [5]", default="5")
            try:
                options['game_ai_workers'] = int(game_workers)
                if options['game_ai_workers'] > 0:
                    break
                print("⚠️  Please enter a positive number")
            except ValueError:
                print("⚠️  Please enter a valid number")

    options['test_mode'] = ask_yes_no("Run in test mode (disable AI calls)?", default=False)

    return options


def build_command(options):
    """Build generate_cache.py command from options dict"""
    # Start with environment variables if configured
    env_vars = []
    if 'team_ai_workers' in options:
        env_vars.append(f"AI_ANALYSIS_WORKERS={options['team_ai_workers']}")
    if 'game_ai_workers' in options:
        env_vars.append(f"GAME_ANALYSIS_WORKERS={options['game_ai_workers']}")

    cmd_parts = []
    if env_vars:
        cmd_parts.extend(env_vars)
    cmd_parts.extend(["python", "generate_cache.py"])

    # Deploy-only mode
    if options.get('deploy_only'):
        cmd_parts.append("--deploy-only")
        # Skip adding other generation flags, jump to deployment
    else:
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

        # AI model override
        if 'ai_model' in options:
            cmd_parts.append(f'--ai-model "{options["ai_model"]}"')

        # Game AI options
        if options.get('skip_game_ai'):
            cmd_parts.append("--skip-game-ai")
        elif 'regenerate_game_ai' in options:
            cmd_parts.append(f'--regenerate-game-ai "{options["regenerate_game_ai"]}"')

        # Dashboard AI options
        if options.get('skip_dashboard_ai'):
            cmd_parts.append("--skip-dashboard-ai")

    # Deployment options
    if options.get('commit'):
        cmd_parts.append("--commit")
    if options.get('deploy_netlify'):
        cmd_parts.append("--deploy-netlify")
    if 'copy_to' in options:
        cmd_parts.append(f'--copy-to "{options["copy_to"]}"')

    # Advanced options
    if options.get('test_mode'):
        cmd_parts.append("--test-mode")

    return " ".join(cmd_parts)


def display_command(cmd):
    """Display command in a nice formatted box"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║  Command to run:" + " " * 61 + "║")
    print("║" + " " * 78 + "║")

    # Word wrap the command if it's too long
    max_width = 74
    words = cmd.split()
    lines = []
    current_line = []

    for word in words:
        # Calculate what the line length would be if we add this word
        test_line = current_line + [word]
        test_length = len(" ".join(test_line))

        if test_length > max_width and current_line:
            # Adding this word would exceed max_width, so finish current line
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            # This word fits
            current_line.append(word)

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
