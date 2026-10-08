from __future__ import print_function

from abaqus import mdb
import os
import sys
import traceback

INPUT_FILE = "35T-flexion-00.inp"
MODEL_NAME = "ImportedGeometry"


def names(repo):
    """Return sorted repository keys without assuming a specific Python version."""
    try:
        return sorted(list(repo.keys()))
    except Exception:
        return []


def print_repo(label, repo, indent=""):
    vals = names(repo)
    print("{}{} ({}): {}".format(indent, label, len(vals), vals))


def main():
    input_path = os.path.abspath(INPUT_FILE)

    print("=" * 78)
    print("Abaqus input-file import test")
    print("Input file: {}".format(input_path))
    print("=" * 78)

    if not os.path.isfile(input_path):
        raise IOError("Input file not found: {}".format(input_path))

    # Remove a same-named model if one already exists.
    if MODEL_NAME in mdb.models:
        del mdb.models[MODEL_NAME]

    print("\nImporting input file with mdb.ModelFromInputFile() ...")
    model = mdb.ModelFromInputFile(
        name=MODEL_NAME,
        inputFileName=input_path
    )
    print("Import completed successfully.")

    print("\nMODEL")
    print_repo("Parts", model.parts)
    print_repo("Materials", model.materials)
    print_repo("Sections", model.sections)
    print_repo("Steps", model.steps)
    print_repo("Interactions", model.interactions)
    print_repo("Interaction properties", model.interactionProperties)
    print_repo("Field output requests", model.fieldOutputRequests)
    print_repo("History output requests", model.historyOutputRequests)

    print("\nPART CONTENTS")
    for part_name in names(model.parts):
        part = model.parts[part_name]
        print("\n  Part: {}".format(part_name))
        try:
            print("    Nodes: {}".format(len(part.nodes)))
        except Exception:
            print("    Nodes: <unavailable>")
        try:
            print("    Elements: {}".format(len(part.elements)))
        except Exception:
            print("    Elements: <unavailable>")

        print_repo("Sets", part.sets, "    ")
        print_repo("Surfaces", part.surfaces, "    ")
        print_repo("Reference points", part.referencePoints, "    ")

    a = model.rootAssembly

    print("\nASSEMBLY")
    print_repo("Instances", a.instances, "  ")
    print_repo("Sets", a.sets, "  ")
    print_repo("Surfaces", a.surfaces, "  ")
    print_repo("Reference points", a.referencePoints, "  ")

    print("\nINSTANCE CONTENTS")
    for inst_name in names(a.instances):
        inst = a.instances[inst_name]
        print("\n  Instance: {}".format(inst_name))
        try:
            print("    Nodes: {}".format(len(inst.nodes)))
        except Exception:
            print("    Nodes: <unavailable>")
        try:
            print("    Elements: {}".format(len(inst.elements)))
        except Exception:
            print("    Elements: <unavailable>")

        # These repositories are present for many imported instances; use
        # getattr so the diagnostic itself does not fail if one is absent.
        print_repo("Sets", getattr(inst, "sets", {}), "    ")
        print_repo("Surfaces", getattr(inst, "surfaces", {}), "    ")
        print_repo("Reference points", getattr(inst, "referencePoints", {}), "    ")

    # Saving a CAE database gives a convenient file you can inspect later.
    cae_path = os.path.abspath("imported_testGeom.cae")
    mdb.saveAs(pathName=cae_path)
    print("\nSaved imported model database:")
    print("  {}".format(cae_path))

    print("\n" + "=" * 78)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 78)
        print("TEST FAILED")
        print("=" * 78)
        traceback.print_exc()
        sys.exit(1)
