import joblib
from interpret import show
import matplotlib.pyplot as plt

# Load the EBM model from a pickle file
model_file = './models/best_ebm_model.pkl'
ebm_model = joblib.load(model_file)

# Get the global explanation for the EBM model
ebm_global = ebm_model.explain_global(name='EBM Feature Importances')

# Render the plot
show(ebm_global, share_tables=True)

# Save the plot to the filesystem
#fig.write_image("ebm_feature_contributions.png")

# Optionally, display the plot inline (if running in an environment that supports it)
# plt.show()
