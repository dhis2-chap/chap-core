import sys

def validate():
    if len(sys.argv) != 4:
        print("UNVALID ARGUMENTS: Usage: ChapProgram.py <dhis2Baseurl> <dhis2Username> <dhis2Password>")
        return False

    return True
