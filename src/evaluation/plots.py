import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def pat_eval_plot(df_combined: pd.DataFrame, x_name: str, y_name: str) -> plt.Figure:
# Assuming train, val, and test dictionaries are defined as described above

# Convert dictionaries to DataFrames
#df_train = pd.DataFrame(train).T
#df_val = pd.DataFrame(val).T
#df_test = pd.DataFrame(test).T

## Combine into a single DataFrame
#df_combined = pd.concat([df_train.add_suffix('_train'),
#                         df_val.add_suffix('_val'),
#                         df_test.add_suffix('_test')], axis=1)

    fig, ax = plt.subplots(figsize=(15, 6))

    ids = df_combined.index
    x = np.arange(len(ids))  # the label locations
    width = 0.25  # the width of the bars

    # Plot train means with std error bars
    ax.errorbar(x, df_combined[f'{x_name}_train'], yerr=df_combined[f'{y_name}_train'], fmt='-o', label='Train', capsize=3)

    # Plot val means with std error bars
    ax.errorbar(x, df_combined[f'{x_name}_val'], yerr=df_combined[f'{y_name}_val'], fmt='-s', label='Validation', capsize=3)

    # Plot test means with std error bars
    ax.errorbar(x, df_combined[f'{x_name}_test'], yerr=df_combined[f'{y_name}_test'], fmt='-^', label='Test', capsize=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('ID')
    ax.set_ylabel('Mean')
    ax.set_title('Mean and Standard Deviation for Train, Validation, and Test Groups')
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90)
    ax.legend()

    fig.tight_layout()
    return fig
    #fig.savefig()
    #plt.show()
