"""
PortReplacer class for applying and restoring port changes.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


class PortReplacer:
    """
    Handles port replacement operations.

    Applies port changes based on configuration.
    """

    def __init__(self, config_path: str, dry_run: bool = False):
        self.config_path = config_path
        self.dry_run = dry_run
        self.changes_log = []
        self.project_root = Path(__file__).parent.parent.parent
        self.changes_log_path = self.project_root / "configs" / "port_changes.json"

    def load_config(self) -> Dict:
        """Load port configuration from YAML file."""
        config_file = self.project_root / self.config_path
        if not config_file.exists():
            print(f"Error: Config file not found: {config_file}")
            sys.exit(1)

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def replace_in_file(self, file_path: Path, old_port: int, new_port: int) -> Tuple[bool, int]:
        """
        Replace old port with new port in a file.

        Uses regex to ensure we only replace standalone port numbers,
        not ports that are part of larger numbers (e.g., 2143 in 12143).
        """
        if not file_path.exists():
            return False, 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False, 0

        old_port_str = str(old_port)
        new_port_str = str(new_port)

        # Use regex to match standalone port numbers (not part of larger numbers)
        # (?<![0-9]) = not preceded by a digit
        # (?![0-9]) = not followed by a digit
        pattern = rf'(?<![0-9]){re.escape(old_port_str)}(?![0-9])'

        # Count matches
        matches = re.findall(pattern, content)
        count = len(matches)

        if count == 0:
            return False, 0

        new_content = re.sub(pattern, new_port_str, content)

        if new_content == content:
            return False, 0

        if not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return False, 0

        return True, count

    def apply_changes(self) -> bool:
        """Apply port number changes according to config."""
        config = self.load_config()
        files_by_port = config.get('files_by_port', {})

        # Get port mappings from config file
        port_mappings = config.get('port_mappings', {})
        instance_prefix = config.get('instance_prefix', '')
        instance_suffix = config.get('instance_suffix', '')

        if not port_mappings:
            print("Error: No port_mappings found in config file")
            sys.exit(1)

        print(f"{'[DRY RUN] ' if self.dry_run else ''}Applying port changes from config...\n")

        total_replacements = 0
        unique_files_changed = set()
        skipped_ports = 0  # Count ports already at target value

        for default_port, port_info in files_by_port.items():
            default_port = int(default_port)
            target_port = port_mappings.get(default_port)

            if target_port is None:
                continue

            if default_port == target_port:
                skipped_ports += 1
                continue

            comment = port_info.get('comment', f'Port {default_port}')
            files = port_info.get('files', [])

            print(f"Port {default_port} → {target_port} ({comment})")

            port_files_changed = 0
            port_replacements = 0

            for file_rel_path in files:
                file_path = self.project_root / file_rel_path
                # Replace default_port with target_port
                changed, count = self.replace_in_file(file_path, default_port, target_port)

                if changed:
                    port_files_changed += 1
                    port_replacements += count
                    unique_files_changed.add(file_rel_path)
                    status = "[DRY RUN] Would change" if self.dry_run else "Changed"
                    print(f"  ✓ {status}: {file_rel_path} ({count} occurrences)")

                    # Record change for restore
                    self.changes_log.append({
                        'file': file_rel_path,
                        'default_port': default_port,
                        'new_port': target_port,
                        'count': count
                    })

            total_replacements += port_replacements
            print(f"  Summary: {port_files_changed} files, {port_replacements} replacements\n")

        print("=" * 70)
        print(f"Total: {len(unique_files_changed)} unique files changed, {total_replacements} replacements")

        # Show skipped ports info
        if skipped_ports > 0:
            print(f"Skipped: {skipped_ports} ports (already at target value)")

        # If no files changed but ports are already at target, this is success (idempotent)
        if len(unique_files_changed) == 0 and skipped_ports > 0:
            print("\n✓ All ports are already at their target values (no changes needed)")

        if not self.dry_run and self.changes_log:
            # Save changes log
            with open(self.changes_log_path, 'w') as f:
                json.dump({
                    'instance_prefix': instance_prefix,
                    'instance_suffix': instance_suffix,
                    'applied_at': datetime.now().isoformat(),
                    'changes': self.changes_log
                }, f, indent=2)
            print(f"\nChanges logged to: {self.changes_log_path}")

        # Success if:
        # - Files were changed, OR
        # - Ports are already at target value (idempotent)
        return len(unique_files_changed) > 0 or skipped_ports > 0

    def restore_from_changelog(self) -> bool:
        """
        Restore ports to default values based on the change log.

        Reads the changelog and reverses all port changes by calling replace_in_file
        with swapped parameters (new_port -> default_port).
        """
        print(f"{'[DRY RUN] ' if self.dry_run else ''}Restoring to default ports based on change log...")

        if not self.changes_log_path.exists():
            print(f"Error: Change log not found: {self.changes_log_path}")
            return False

        try:
            with open(self.changes_log_path, 'r') as f:
                log_data = json.load(f)
        except Exception as e:
            print(f"Error reading change log: {e}")
            return False

        changes = log_data.get('changes', [])
        if not changes:
            print("No changes recorded in log file")
            return True

        # Group changes by file
        files_to_restore: Dict[str, list] = {}
        for change in changes:
            file_path = change['file']
            if file_path not in files_to_restore:
                files_to_restore[file_path] = []
            files_to_restore[file_path].append(change)

        print(f"Restoring {len(files_to_restore)} files to default ports...\n")

        total_files_restored = 0
        total_replacements = 0
        errors = []

        for file_rel_path, file_changes in files_to_restore.items():
            file_path = self.project_root / file_rel_path

            if not file_path.exists():
                print(f"  ⚠ File not found (skipped): {file_rel_path}")
                continue

            file_replacement_count = 0

            for change in file_changes:
                default_port = change['default_port']
                new_port = change['new_port']

                # Use replace_in_file but swap parameters: new_port -> default_port
                changed, count = self.replace_in_file(file_path, new_port, default_port)

                if changed:
                    file_replacement_count += count

            if file_replacement_count > 0:
                total_files_restored += 1
                total_replacements += file_replacement_count
                status = "[DRY RUN] Would restore" if self.dry_run else "Restored"
                print(f"  ✓ {status}: {file_rel_path} ({file_replacement_count} replacements)")

        if errors:
            print(f"\n⚠ Errors encountered ({len(errors)}):")
            for error in errors:
                print(f"  ✗ {error}")
            return False

        print(f"\n✓ Restored {total_files_restored} files, {total_replacements} replacements")
        return True

    def check_status(self) -> None:
        """
        Check current port status by reading the changelog and verifying files.

        Shows which ports have been changed and validates that the changes
        were applied correctly (old port replaced, new port present).
        """
        if not self.changes_log_path.exists():
            print("No change log found.")
            print("Ports are at default values (no changes applied).")
            return

        try:
            with open(self.changes_log_path, 'r') as f:
                log_data = json.load(f)
        except Exception as e:
            print(f"Error reading change log: {e}")
            return

        changes = log_data.get('changes', [])
        applied_at = log_data.get('applied_at', 'unknown')

        print("=" * 70)
        print("PORT STATUS")
        print("=" * 70)
        print(f"\nChange log found: {self.changes_log_path.name}")
        print(f"Applied at: {applied_at}")
        print(f"\nPort changes:")

        # Group changes by port
        port_changes: Dict[Tuple[int, int], List[str]] = {}
        for change in changes:
            default_port = change['default_port']
            new_port = change['new_port']
            file_path = change['file']

            key = (default_port, new_port)
            if key not in port_changes:
                port_changes[key] = []
            port_changes[key].append(file_path)

        all_ok = True

        for (default_port, new_port), files in sorted(port_changes.items()):
            if default_port == new_port:
                print(f"\n  {default_port} → {new_port} (no change)")
                print(f"    {len(files)} files (skipped)")
                continue

            print(f"\n  {default_port} → {new_port}")
            print(f"    Files: {len(files)}")

            # Verify a sample file to check if replacement was successful
            sample_file = self.project_root / files[0]
            if sample_file.exists():
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check if old port still exists
                    old_pattern = rf'(?<![0-9]){default_port}(?![0-9])'
                    old_exists = bool(re.search(old_pattern, content))

                    # Check if new port exists
                    new_pattern = rf'(?<![0-9]){new_port}(?![0-9])'
                    new_exists = bool(re.search(new_pattern, content))

                    if old_exists and not new_exists:
                        print(f"    ⚠ WARNING: Still has old port {default_port}, missing new port {new_port}")
                        all_ok = False
                    elif not old_exists and new_exists:
                        print(f"    ✓ OK: New port {new_port} applied")
                    elif old_exists and new_exists:
                        print(f"    ⚠ WARNING: Both old ({default_port}) and new ({new_port}) ports found")
                        all_ok = False
                    else:
                        print(f"    ⚠ WARNING: Neither port found in sample file")
                        all_ok = False

                except Exception as e:
                    print(f"    ⚠ Error checking file: {e}")
                    all_ok = False
            else:
                print(f"    ⚠ Sample file not found: {files[0]}")

        print("\n" + "=" * 70)
        if all_ok:
            print("✓ All port changes applied successfully")
        else:
            print("⚠ Some issues detected - see warnings above")
        print("=" * 70)
