import os
import re
import pickle
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D


# --- Helper Functions ---
def first_derivative(data):
    return savgol_filter(data, 25, polyorder=5, deriv=1, axis=1)


def second_derivative(data):
    return savgol_filter(data, 31, polyorder=4, deriv=2, axis=1)


def SNV(data):
    return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)


def custom_sort_key(label):
    # This regex matches prefix (non-digits) and trailing number
    match = re.match(r'([^\d]+)(\d+)$', label.replace('_', '').replace(' ', ''))
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
    else:
        # If no number at the end, treat whole label as prefix, number as large
        prefix = label
        number = float('inf')
    return (prefix, number)

def average_by_label(data_matrix, label_list):
    df = pd.DataFrame(data_matrix)
    df['label'] = label_list
    averaged = df.groupby('label').mean().reindex(sorted(df['label'].unique(), key=custom_sort_key))
    return averaged.values


def plot_feature_avg(data_matrix, label_list, title, _, wavenumbers):
    averaged = average_by_label(data_matrix, label_list)
    fig, ax = plt.subplots()
    unique_labels = sorted(set(label_list), key=custom_sort_key)
    for i in range(averaged.shape[0]):
        plt.plot(wavenumbers, averaged[i], label=unique_labels[i])
    plt.xlabel('Raman Shift (cm-1)')
    plt.ylabel('Intensity')
    plt.title(title)
    ax.invert_xaxis()
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# --- Main Application ---
class PCAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chemometrics PCA Tool")

        self.data = None
        self.wavenumbers = None
        self.lab = None
        self.new_data = None
        self.new_labels = None

        style = ttkb.Style()
        style.configure("Custom.TButton", foreground="#0ee8c7")

        self.file_label = ttkb.Label(root, text="No file selected", bootstyle="info")
        self.file_label.pack(pady=5)

        ttkb.Button(root, text="Select CSV File", command=self.load_csv, bootstyle="info-outline").pack(pady=5)

        self.feature_label = ttkb.Label(root, text="Select Data Type", bootstyle="info")
        self.feature_label.pack()
        self.feature_dropdown = ttkb.Combobox(root, state="readonly", bootstyle="info")
        self.feature_dropdown.pack(pady=5)

        self.deriv_label = ttkb.Label(root, text="Savitzky-Golay Derivative (PCA only):", bootstyle="info")
        self.deriv_label.pack()
        self.deriv_var = ttkb.StringVar(value="none")
        ttkb.Radiobutton(root, text="None", variable=self.deriv_var, value="none", bootstyle="info-toolbutton").pack(
            anchor="center")
        ttkb.Radiobutton(root, text="1st Derivative", variable=self.deriv_var, value="1st",
                         bootstyle="info-toolbutton").pack(anchor="center")
        ttkb.Radiobutton(root, text="2nd Derivative", variable=self.deriv_var, value="2nd",
                         bootstyle="info-toolbutton").pack(anchor="center")

        ttkb.Button(root, text="Plot Data Sets", command=self.plot_all, bootstyle="info-outline").pack(pady=5)

        # Updated buttons for separate functionality
        ttkb.Button(root, text="Run PCA", command=self.run_pca, bootstyle="info-outline").pack(pady=5)
        ttkb.Button(root, text="Plot PCA", command=self.plot_pca, bootstyle="info-outline").pack(pady=5)

        ttkb.Button(root, text="Export PCA to Excel", command=self.export_excel, bootstyle="info-outline").pack(pady=5)

        ttkb.Button(root, text="Add New Dataset", command=self.add_new_dataset, style="Custom.TButton").pack(pady=5)
        ttkb.Button(root, text="Replot PCA with New Data", command=self.replot_new_data, style="Custom.TButton").pack(
            pady=5)

        ttkb.Button(root, text="Save PCA Session", command=self.save_session, style="Custom.TButton").pack(pady=5)
        ttkb.Button(root, text="Load PCA Session", command=self.load_session, style="Custom.TButton").pack(pady=5)
        ttkb.Button(root, text="Clear Data", command=self.clear_data, bootstyle="warning-outline").pack(pady=5)
        ttkb.Button(root, text="Exit", command=root.quit, bootstyle="danger-outline").pack(pady=5)

        self.features = {}
        self.current_feat = None

    def clear_data(self):
        # Reset all internal variables
        self.data = None
        self.wavenumbers = None
        self.lab = None
        self.new_data = None
        self.new_labels = None
        self.features = {}
        self.current_feat = None

        # Clear dropdown menu
        self.feature_dropdown.set('')
        self.feature_dropdown["values"] = []

        # Clear file label
        self.file_label.config(text="No file selected")

        # Reset derivative selection
        self.deriv_var.set("none")

        # Remove PCA results if any
        if hasattr(self, 'pca_model'):
            del self.pca_model
        if hasattr(self, 'Xt'):
            del self.Xt

        messagebox.showinfo("Data Cleared", "All data has been cleared. You can start fresh.")

    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel files", "*.xlsx *.xls")])
        if not filepath:
            return

        self.file_label.config(text=os.path.basename(filepath))
        # Determine file type and load accordingly
        _, ext = os.path.splitext(filepath)
        if ext.lower() == '.csv':
            df = pd.read_csv(filepath, header=None).T
        elif ext.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, header=None).T
        else:
            raise ValueError("Unsupported file type selected.")
        self.wavenumbers = df.iloc[0, 1:].astype(float).values
        self.lab = df.iloc[1:, 0].str[:-2].tolist()
        feat = df.iloc[1:, 1:].astype(float).values

        snv = SNV(feat)

        self.features = {
            "Raw absorbance data": feat,
            "SNV": snv
        }
        self.feature_dropdown["values"] = list(self.features.keys())
        self.feature_dropdown.current(0)

        self.feat = feat
        self.snv = snv
        self.feat_d = first_derivative(feat)
        self.snv_d = first_derivative(snv)
        self.snv_sd = second_derivative(snv)

    def preprocess(self, data):
        if self.deriv_var.get() == "1st":
            return first_derivative(data)
        elif self.deriv_var.get() == "2nd":
            return second_derivative(data)
        return data

    def align_features(self, ref, new):
        diff = new.shape[1] - ref.shape[1]
        if diff > 0:
            new = new[:, :-diff]
        elif diff < 0:
            last = new[:, -1][:, None]
            new = np.hstack([new, np.repeat(last, -diff, axis=1)])
        return new

    def add_new_dataset(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel files", "*.xlsx *.xls")])
        if not path:
            return

        # Determine file type and load accordingly
        _, ext = os.path.splitext(path)
        if ext.lower() == '.csv':
            df = pd.read_csv(path, header=None).T
        elif ext.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path, header=None).T
        else:
            raise ValueError("Unsupported file type selected.")
        labels = ["New: " + lbl for lbl in df.iloc[1:, 0].str[:-2].tolist()]
        new_feat = df.iloc[1:, 1:].astype(float).values

        # Get the original selected data type (Raw, SNV, etc.)
        selected = self.features[self.feature_dropdown.get()]

        # Align the features of the new dataset with the selected feature set
        new_feat = self.align_features(selected, new_feat)

        # Preprocess the new data using the same method applied to the original dataset
        if self.feature_dropdown.get() == "SNV":
            new_feat = SNV(new_feat)

        new_feat = self.preprocess(new_feat)

        self.new_data = new_feat
        self.new_labels = labels

        messagebox.showinfo("New Dataset Loaded", "New dataset loaded and preprocessed.")


    def plot_all(self):
        for title, data in zip(
                ["Raw Data (avg)", "SNV (avg)", "1st Derivative (avg)", "SNV + 1st Derivative (avg)",
                 "SNV + 2nd Derivative (avg)"],
                [self.feat, self.snv, self.feat_d, self.snv_d, self.snv_sd]
        ):
            ymin, ymax = np.min(data), np.max(data)
            plot_feature_avg(data, self.lab, title, (ymin, ymax), self.wavenumbers)

    def run_pca(self):
        data = self.features[self.feature_dropdown.get()]
        data = self.preprocess(data)

        model = PCA(n_components=5, svd_solver='full', random_state=42)
        Xt = model.fit_transform(data)
        self.pca_model = model
        self.Xt = Xt

        unique = sorted(set(self.lab), key=custom_sort_key)
        colors = [plt.cm.jet(i / len(unique)) for i in range(len(unique))]

        # Store PCA data, but do not plot it yet
        messagebox.showinfo("PCA Computed", "PCA data has been computed and stored.")

    def plot_pca(self):
        if not hasattr(self, 'Xt'):
            messagebox.showwarning("No PCA Data", "Please run PCA first to generate data.")
            return

        # Ensure 'lab' exists and is properly assigned
        if not hasattr(self, 'lab'):
            self.lab = np.zeros(self.Xt.shape[0])  # Default to no labels if not provided

        # Sort unique labels and assign colors accordingly
        unique = sorted(set(self.lab), key=custom_sort_key)
        colors = [plt.cm.jet(i / len(unique)) for i in range(len(unique))]

        # Create figure for 3D and 2D plots
        fig = plt.figure(figsize=(18, 6))

        # 3D plot
        ax3d = fig.add_subplot(131, projection='3d')
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        # Plot for each unique label
        for i, u in enumerate(unique):
            idx = [j for j in range(len(self.lab)) if self.lab[j] == u]
            ax3d.scatter(*self.Xt[idx].T, c=[colors[i]], s=60, edgecolors='k', label=u, marker='o')
            ax1.scatter(self.Xt[idx, 0], self.Xt[idx, 1], c=[colors[i]], s=60, edgecolors='k', marker='o')
            ax2.scatter(self.Xt[idx, 0], self.Xt[idx, 2], c=[colors[i]], s=60, edgecolors='k', marker='o')

        # Set titles and labels
        ax3d.set_title('PCA 3D Plot')
        ax3d.set_xlabel('PC1');
        ax3d.set_ylabel('PC2');
        ax3d.set_zlabel('PC3')
        ax1.set_title("PC1 vs PC2")
        ax1.set_xlabel("PC1");
        ax1.set_ylabel("PC2")
        ax2.set_title("PC1 vs PC3")
        ax2.set_xlabel("PC1");
        ax2.set_ylabel("PC3")

        # Add legend in the upper left
        ax3d.legend(loc='upper left', fontsize='small')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def replot_new_data(self):
        if not hasattr(self, 'Xt') or not hasattr(self, 'pca_model'):
            messagebox.showwarning("Missing PCA", "Please run PCA on the original dataset first.")
            return
        if self.new_data is None or self.new_labels is None:
            messagebox.showwarning("Missing New Data", "Please add a new dataset before replotting.")
            return

        # Project new data using existing PCA model
        new_proj = self.pca_model.transform(self.new_data)

        # Combine old and new projections for plotting
        all_proj = np.vstack([self.Xt, new_proj])
        all_labels = self.lab + self.new_labels

        # Separate original and new labels
        unique_orig = sorted(set(self.lab), key=custom_sort_key)
        unique_new = sorted(set(self.new_labels), key=custom_sort_key)

        # Build color maps
        color_map_orig = {label: plt.cm.jet(i / len(unique_orig)) for i, label in enumerate(unique_orig)}
        color_map_new = {label: plt.cm.Set2(i / len(unique_new)) for i, label in enumerate(unique_new)}

        # Create plots
        fig = plt.figure(figsize=(18, 6))
        ax3d = fig.add_subplot(131, projection='3d')
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        # Plot original data with circle markers
        for label in unique_orig:
            idx = [i for i, l in enumerate(self.lab) if l == label]
            ax3d.scatter(*self.Xt[idx].T, c=[color_map_orig[label]], s=60, edgecolors='k', label=label, marker='o')
            ax1.scatter(self.Xt[idx, 0], self.Xt[idx, 1], c=[color_map_orig[label]], s=60, edgecolors='k', label=label,
                        marker='o')
            ax2.scatter(self.Xt[idx, 0], self.Xt[idx, 2], c=[color_map_orig[label]], s=60, edgecolors='k', label=label,
                        marker='o')

        # Plot new data with triangle markers
        for label in unique_new:
            idx = [i for i, l in enumerate(self.new_labels) if l == label]
            proj_idx = [i + len(self.Xt) for i in idx]  # offset to index from combined projection
            ax3d.scatter(*all_proj[proj_idx].T, c=[color_map_new[label]], s=60, edgecolors='k', label=label, marker='^')
            ax1.scatter(all_proj[proj_idx, 0], all_proj[proj_idx, 1], c=[color_map_new[label]], s=60, edgecolors='k',
                        label=label, marker='^')
            ax2.scatter(all_proj[proj_idx, 0], all_proj[proj_idx, 2], c=[color_map_new[label]], s=60, edgecolors='k',
                        label=label, marker='^')

        ax3d.set_title("3D PCA Plot")
        ax1.set_title("PC1 vs PC2")
        ax2.set_title("PC1 vs PC3")

        for ax in [ax1, ax2, ax3d]:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2" if ax != ax2 else "PC3")

        # Sort legend alphabetically/numerically
        handles, labels = ax1.get_legend_handles_labels()
        sorted_labels = sorted(zip(labels, handles), key=lambda x: custom_sort_key(x[0]))
        ax1.legend([h for _, h in sorted_labels], [l for l, _ in sorted_labels], loc='upper left')

        plt.tight_layout()
        plt.show()

    def export_excel(self):
        if not hasattr(self, "Xt") or self.Xt is None or not hasattr(self, "lab") or self.lab is None:
            messagebox.showerror("Error", "No PCA data or sample labels to export.")
            return

        # Check if new data is present and projected
        has_new = hasattr(self, "new_data") and self.new_data is not None \
                  and hasattr(self, "new_labels") and self.new_labels is not None \
                  and hasattr(self, "pca_model")

        export_file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if not export_file_path:
            return

        try:
            # Original data PCA scores and labels
            orig_df = pd.DataFrame({
                "PC1": self.Xt[:, 0],
                "PC2": self.Xt[:, 1],
                "PC3": self.Xt[:, 2],
                "PC4": self.Xt[:, 3],
                "PC5": self.Xt[:, 4],
                "Sample Names": self.lab,
                "Data Type": "Original"
            })

            if has_new:
                # Project new data if not already projected (just to be safe)
                new_proj = self.pca_model.transform(self.new_data)
                new_df = pd.DataFrame({
                    "PC1": new_proj[:, 0],
                    "PC2": new_proj[:, 1],
                    "PC3": new_proj[:, 2],
                    "Sample Names": self.new_labels,
                    "Data Type": "New"
                })

                # Combine original and new PCA data
                combined_df = pd.concat([orig_df, new_df], ignore_index=True)
            else:
                combined_df = orig_df

            # Prepare Preprocessing Settings
            settings_dict = {
                "Selected Feature Set": self.feature_dropdown.get(),
                "Derivative Applied": self.deriv_var.get()
            }

            # Write to Excel
            with pd.ExcelWriter(export_file_path, engine="openpyxl") as writer:
                combined_df.to_excel(writer, sheet_name="PCA Data", index=False)

                settings_df = pd.DataFrame(list(settings_dict.items()), columns=["Setting", "Value"])
                settings_df.to_excel(writer, sheet_name="Preprocessing Settings", index=False)

            messagebox.showinfo("Export Successful", f"PCA data and settings have been exported to {export_file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during export: {e}")

    def save_session(self):
        if not hasattr(self, 'Xt'):
            messagebox.showwarning("Warning", "Please run PCA first.")
            return

        if self.new_data is not None:
            self.lab += [lbl.replace("New: ", "") for lbl in self.new_labels]
            base_data = self.features[self.feature_dropdown.get()]
            base_data = self.preprocess(base_data)
            merged = np.vstack([base_data, self.new_data])
            self.features[self.feature_dropdown.get()] = merged
            self.new_data, self.new_labels = None, None

        session_data = {
            'model': self.pca_model,
            'projected': self.Xt,
            'labels': self.lab,
            'feature_name': self.feature_dropdown.get(),
            'derivative': self.deriv_var.get(),
            'wavenumbers': self.wavenumbers,
            'features': self.features
        }

        path = filedialog.asksaveasfilename(defaultextension=".pca", filetypes=[("PCA Session", "*.pca")])
        if path:
            with open(path, 'wb') as f:
                pickle.dump(session_data, f)
            messagebox.showinfo("Session Saved", f"Session saved to {path}")

    def load_session(self):
        path = filedialog.askopenfilename(filetypes=[("PCA Session", "*.pca")])
        if not path:
            return

        with open(path, 'rb') as f:
            session = pickle.load(f)

        self.pca_model = session['model']
        self.Xt = session['projected']
        self.lab = session['labels']
        self.wavenumbers = session['wavenumbers']
        self.features = session['features']

        self.feature_dropdown["values"] = list(self.features.keys())
        self.feature_dropdown.set(session['feature_name'])
        self.deriv_var.set(session['derivative'])

        messagebox.showinfo("Session Loaded", "PCA session loaded successfully.")

# Main execution
root = ttkb.Window(themename="flatly")
app = PCAApp(root)
root.mainloop()
