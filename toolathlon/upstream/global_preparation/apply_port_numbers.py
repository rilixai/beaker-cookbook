#!/usr/bin/env python3
"""
Apply port number changes across the Toolathlon codebase.

This script manages port configurations based on the port_mappings
defined in configs/ports_config.yaml.

Usage:
    # Apply port changes using config file
    python global_preparation/apply_port_numbers.py
    python global_preparation/apply_port_numbers.py --dry-run

    # Restore to default ports
    python global_preparation/apply_port_numbers.py --restore

    # Check current port status
    python global_preparation/apply_port_numbers.py --status

    # Skip confirmation prompt (auto yes)
    python global_preparation/apply_port_numbers.py -y

State Machine:
    1. Initial state: Default ports, no changelog
    2. After apply: Modified ports, changelog exists
    3. Apply again: Restores to default first, then applies new config
    4. After restore: Default ports, no changelog (back to initial)
"""

import argparse
import sys

from multiple_instance import PortReplacer


def main():
    """Main entry point for the port management script."""
    parser = argparse.ArgumentParser(
        description='Apply port number changes for Toolathlon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply using port_mappings from config file
  python global_preparation/apply_port_numbers.py
  python global_preparation/apply_port_numbers.py --dry-run
  python global_preparation/apply_port_numbers.py -y  # auto yes

  # Restore to default ports
  python global_preparation/apply_port_numbers.py --restore
  python global_preparation/apply_port_numbers.py --restore -y  # auto yes

  # Check current port status
  python global_preparation/apply_port_numbers.py --status

State Machine:
  Initial: Default ports, no changelog
  After apply: Modified ports, changelog exists
  Next apply: Restores first, then applies new changes
  After restore: Default ports, no changelog (back to initial)
        """
    )

    parser.add_argument(
        '--config',
        default='configs/ports_config.yaml',
        help='Path to ports config file (default: configs/ports_config.yaml)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would change without modifying files'
    )

    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Check current port status'
    )

    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore ports to default values (from changelog)'
    )

    args = parser.parse_args()

    # Handle --status
    if args.status:
        replacer = PortReplacer(args.config)
        replacer.check_status()
        sys.exit(0)

    # Handle --restore
    if args.restore:
        _handle_restore(args.config, args.dry_run, args.yes)
        sys.exit(0)

    # Default: apply changes
    _handle_apply(args.config, args.dry_run, args.yes)


def _handle_apply(config_path: str, dry_run: bool, auto_yes: bool) -> None:
    """Handle apply command."""
    replacer = PortReplacer(config_path, dry_run=dry_run)
    has_changelog = replacer.changes_log_path.exists()

    print("=" * 70)
    print("PORT CHANGE OPERATION")
    print("=" * 70)

    if has_changelog:
        print("\n📄 Found existing change log")
        print("   This means ports have been modified before.")
        print("\n🔄 Operation plan:")
        print("   1. Restore to default ports (using change log)")
        print("   2. Apply new port mappings (from config)")
    else:
        print("\n📄 No change log found")
        print("   This should be the FIRST time you're changing ports.")
        print("\n🔄 Operation plan:")
        print("   1. Apply port mappings (from config)")

    print("\n" + "=" * 70)

    # Ask for confirmation
    if not auto_yes and not dry_run:
        response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Aborted.")
            sys.exit(0)

    print()

    # Step 1: If changelog exists, restore to default first
    if has_changelog:
        print("STEP 1: Restoring to default ports")
        print("-" * 70)
        success = replacer.restore_from_changelog()
        if not success:
            print("\n❌ Failed to restore ports")
            sys.exit(1)

        # Delete changelog after successful restore
        if not dry_run:
            try:
                replacer.changes_log_path.unlink()
                print(f"✓ Removed change log: {replacer.changes_log_path.name}")
            except Exception as e:
                print(f"⚠ Warning: Could not remove change log: {e}")

        print()
        print("STEP 2: Applying new port mappings")
        print("-" * 70)
    else:
        print("STEP 1: Applying port mappings")
        print("-" * 70)

    # Step 2: Apply new changes
    success = replacer.apply_changes()

    if not success and not dry_run:
        print("\n❌ Failed to apply port changes")
        sys.exit(1)

    print("\n" + "=" * 70)
    if dry_run:
        print("✓ Dry-run completed")
    else:
        print("✓ Operation completed successfully")
    print("=" * 70)


def _handle_restore(config_path: str, dry_run: bool, auto_yes: bool) -> None:
    """Handle restore command."""
    replacer = PortReplacer(config_path, dry_run=dry_run)

    if not replacer.changes_log_path.exists():
        print("=" * 70)
        print("RESTORE OPERATION")
        print("=" * 70)
        print("\nNo change log found.")
        print("Ports are already at default values (no changes to restore).")
        print("=" * 70)
        sys.exit(0)

    print("=" * 70)
    print("RESTORE OPERATION")
    print("=" * 70)
    print("\n📄 Found change log")
    print("   Will restore ports to default values.")
    print("\n🔄 Operation plan:")
    print("   1. Restore to default ports (using change log)")
    print("   2. Delete change log")
    print("\n" + "=" * 70)

    # Ask for confirmation
    if not auto_yes and not dry_run:
        response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Aborted.")
            sys.exit(0)

    print()

    # Restore to default
    print("Restoring to default ports")
    print("-" * 70)
    success = replacer.restore_from_changelog()

    if not success:
        print("\n❌ Failed to restore ports")
        sys.exit(1)

    # Delete changelog after successful restore
    if not dry_run:
        try:
            replacer.changes_log_path.unlink()
            print(f"\n✓ Removed change log: {replacer.changes_log_path.name}")
        except Exception as e:
            print(f"\n⚠ Warning: Could not remove change log: {e}")

    print("\n" + "=" * 70)
    if dry_run:
        print("✓ Dry-run completed")
    else:
        print("✓ Restore completed successfully")
    print("=" * 70)


if __name__ == '__main__':
    main()
