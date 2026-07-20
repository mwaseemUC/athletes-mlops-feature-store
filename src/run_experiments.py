"""Step 4 driver: run all four experiment combinations.
2 feature versions x 2 hyperparameter configs, same algorithm throughout.
"""
import train

FEATURE_VERSIONS = ["v1", "v2"]
C_VALUES = [0.1, 10.0]


def main():
    results = []
    for version in FEATURE_VERSIONS:
        for C in C_VALUES:
            m = train.run(version, C)
            results.append({"version": version, "C": C, **m})

    print("\n==== experiment summary ====")
    header = f"{'version':<8}{'C':<8}{'accuracy':<10}{'f1':<10}{'roc_auc':<10}"
    print(header)
    for r in results:
        print(f"{r['version']:<8}{r['C']:<8}{r['accuracy']:<10.4f}"
              f"{r['f1']:<10.4f}{r['roc_auc']:<10.4f}")


if __name__ == "__main__":
    main()