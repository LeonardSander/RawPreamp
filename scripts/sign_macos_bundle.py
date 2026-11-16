#!/usr/bin/env python3
"""Sign bundle contents before their enclosing code objects."""

import argparse
from pathlib import Path
import plistlib
import subprocess

from validate_macos_bundle import MACHO_MAGIC


def signing_order(app):
    app = app.resolve()
    images = []
    for path in (app / "Contents").rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                if stream.read(4) in MACHO_MAGIC:
                    images.append(path)

    # Discover enclosing code bundles from actual Mach-O images so resource
    # directories with .bundle suffixes are not treated as executable code.
    containers = {app}
    for image in images:
        containers.update(parent for parent in image.parents
                          if parent.is_relative_to(app) and parent.suffix in
                          {".framework", ".app", ".xpc", ".appex", ".bundle"})

    # A bundle's main executable can cause codesign to inspect its nested
    # code. Sign it through the bundle itself, after all descendants. Plain
    # dylibs and plugins are signed individually first.
    units = set(images) | containers
    for image in images:
        if image.parent.name == "MacOS" and image.parent.parent.name == "Contents":
            owner = image.parent.parent.parent
            if owner in containers:
                # Helpers are independent code, while the bundle executable
                # is signed by codesign's enclosing-bundle operation below.
                # Read the declared executable rather than guessing its name.
                plist = owner / "Contents/Info.plist"
                if plist.is_file():
                    with plist.open("rb") as stream:
                        executable = plistlib.load(stream).get("CFBundleExecutable")
                    if image.name == executable:
                        units.discard(image)

    return sorted(units, key=lambda path: (-len(path.parts), str(path)))


def sign(app):
    for path in signing_order(app):
        print(f"Signing: {path}", flush=True)
        subprocess.run(["codesign", "--force", "--sign", "-", str(path)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", "--verbose=2",
                    str(app.resolve())], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    sign(parser.parse_args().app)
