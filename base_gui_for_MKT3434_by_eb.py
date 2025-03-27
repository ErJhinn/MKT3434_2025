
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QTextEdit, QScrollArea, QStatusBar, QProgressBar, QDialog, QMessageBox, QFileDialog, QLineEdit
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from sklearn import datasets, model_selection, preprocessing, metrics
from sklearn.impute import SimpleImputer  


from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor, QuantileRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, log_loss, hinge_loss



class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        # Main container
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Initialize data containers
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.current_model = None

        # Create UI sections
        self.create_data_section()        
        self.create_tabs()                
        self.create_visualization()      
        self.create_status_bar()          


    

    def create_data_section(self):
        """Create the dataset selection and preprocessing options section at the top."""
        data_group = QGroupBox("Dataset and Preprocessing")
        data_layout = QHBoxLayout()

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)

        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["No Scaling", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"])

       
        self.missing_combo = QComboBox()
        self.missing_combo.addItems([
            "No Imputation",
            "Mean Imputation",
            "Median Imputation",
            "Most Frequent Imputation",
            "Constant Imputation"
        ])

        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)

        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Missing Values:"))
        data_layout.addWidget(self.missing_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)


    # Tabs Section - Updated with Regression & Classification Tabs
    def create_tabs(self):
        from PyQt6.QtWidgets import QTabWidget
        self.tab_widget = QTabWidget()

        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_content = create_func()
            scroll.setWidget(tab_content)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        self.layout.addWidget(self.tab_widget)


    # Classical ML Tab with Regression and Classification Sections

    def create_classical_ml_tab(self):
        from PyQt6.QtWidgets import QStackedWidget
        widget = QWidget()
        layout = QGridLayout(widget)

       # Regression Section
        regression_group = QGroupBox("Regression")
        reg_layout = QVBoxLayout()

        reg_model_combo = QComboBox()
        reg_model_combo.addItems(["Linear Regression", "Decision Tree Regressor", "Support Vector Regressor"])
        reg_layout.addWidget(QLabel("Model:"))
        reg_layout.addWidget(reg_model_combo)

        # Linear Regression parameters with loss options
        lr_params = {
            "fit_intercept": "checkbox",
            "normalize": "checkbox",
            "loss_function": ["MSE", "MAE", "Huber"]
        }
        lr_group = self.create_algorithm_group("Linear Regression", lr_params)

        # Decision Tree Regressor parameters with loss function selection
        dt_reg_params = {
            "max_depth": "int",
            "min_samples_split": "int",
            "loss_function": ["MSE", "MAE"]
        }
        dt_group = self.create_algorithm_group("Decision Tree Regressor", dt_reg_params)

        # Support Vector Regressor parameters with kernel and hyperparameters
        svr_params = {
            "C": "double",
            "epsilon": "double",
            "kernel": ["linear", "rbf", "poly"],
            "degree": "int"
        }
        svr_group = self.create_algorithm_group("Support Vector Regressor", svr_params)

        reg_stack = QStackedWidget()
        reg_stack.addWidget(lr_group)
        reg_stack.addWidget(dt_group)
        reg_stack.addWidget(svr_group)
        reg_layout.addWidget(reg_stack)
        reg_model_combo.currentIndexChanged.connect(reg_stack.setCurrentIndex)
        reg_stack.setCurrentIndex(0)

        svr_kernel_combo = svr_group.findChild(QComboBox, "kernel")
        svr_degree_spin = svr_group.findChild(QSpinBox, "degree")
        if svr_kernel_combo and svr_degree_spin:
            def on_svr_kernel_changed(kernel_text):
                svr_degree_spin.setDisabled(kernel_text != "poly")
            svr_kernel_combo.currentTextChanged.connect(on_svr_kernel_changed)
            on_svr_kernel_changed(svr_kernel_combo.currentText())

        regression_group.setLayout(reg_layout)
        layout.addWidget(regression_group, 0, 0)

        #  Classification Section
        classification_group = QGroupBox("Classification")
        class_layout = QVBoxLayout()

        class_model_combo = QComboBox()
        class_model_combo.addItems([
            "Logistic Regression", "Decision Tree", "Random Forest",
            "Support Vector Machine", "Naive Bayes", "K-Nearest Neighbors"
        ])
        class_layout.addWidget(QLabel("Model:"))
        class_layout.addWidget(class_model_combo)

        # Logistic Regression parameters with loss function options
        logit_params = {
            "C": "double",
            "max_iter": "int",
            "multi_class": ["ovr", "multinomial"],
            "loss_function": ["Cross-Entropy", "Hinge"]
        }
        logit_group = self.create_algorithm_group("Logistic Regression", logit_params)

        # Decision Tree Classifier parameters
        dt_clf_params = {
            "max_depth": "int",
            "min_samples_split": "int",
            "criterion": ["gini", "entropy"]
        }
        dtc_group = self.create_algorithm_group("Decision Tree", dt_clf_params)

        # Random Forest Classifier parameters
        rf_params = {
            "n_estimators": "int",
            "max_depth": "int",
            "min_samples_split": "int"
        }
        rf_group = self.create_algorithm_group("Random Forest", rf_params)


        svm_params = {
            "C": "double",
            "kernel": ["linear", "rbf", "poly"],
            "degree": "int",
            "loss_function": ["Hinge", "Cross-Entropy"]
        }
        svm_group = self.create_algorithm_group("Support Vector Machine", svm_params)


        nb_params = {
            "var_smoothing": "double",
            "priors": ["Data-Derived", "Uniform", "Custom"],  
            "custom_priors": ("text", "[0.3, 0.7]")            
        }
        nb_group = self.create_algorithm_group("Naive Bayes", nb_params)

        # K-Nearest Neighbors parameters
        knn_params = {
            "n_neighbors": "int",
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
        knn_group = self.create_algorithm_group("K-Nearest Neighbors", knn_params)

        class_stack = QStackedWidget()
        class_stack.addWidget(logit_group)
        class_stack.addWidget(dtc_group)
        class_stack.addWidget(rf_group)
        class_stack.addWidget(svm_group)
        class_stack.addWidget(nb_group)
        class_stack.addWidget(knn_group)
        class_layout.addWidget(class_stack)
        class_model_combo.currentIndexChanged.connect(class_stack.setCurrentIndex)
        class_stack.setCurrentIndex(0)


        svm_kernel_combo = svm_group.findChild(QComboBox, "kernel")
        svm_degree_spin = svm_group.findChild(QSpinBox, "degree")
        svm_loss_combo = svm_group.findChild(QComboBox, "loss_function")
        if svm_kernel_combo and svm_degree_spin and svm_loss_combo:
            def on_svm_kernel_changed(kernel_text):
                svm_degree_spin.setDisabled(kernel_text != "poly")
                if kernel_text != "linear":
                    if svm_loss_combo.currentText() == "Cross-Entropy":
                        svm_loss_combo.setCurrentText("Hinge")
                    svm_loss_combo.setDisabled(True)
                else:
                    svm_loss_combo.setDisabled(False)
            svm_kernel_combo.currentTextChanged.connect(on_svm_kernel_changed)
            on_svm_kernel_changed(svm_kernel_combo.currentText())

        classification_group.setLayout(class_layout)
        layout.addWidget(classification_group, 0, 1)

        return widget


    # Dimensionality Reduction, Deep Learning, and RL Tabs (Skeletons)

    def create_dim_reduction_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        kmeans_params = self.create_algorithm_group("K-Means Parameters", {
            "n_clusters": "int",
            "max_iter": "int",
            "n_init": "int"
        })
        kmeans_layout.addWidget(kmeans_params)
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        pca_params = self.create_algorithm_group("PCA Parameters", {
            "n_components": "int",
            "whiten": "checkbox"
        })
        pca_layout.addWidget(pca_params)
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        return widget

    def create_deep_learning_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        # (Placeholder for Deep Learning UI)
        return widget

    def create_rl_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        # (Placeholder for Reinforcement Learning UI)
        return widget

    # Visualization Panel for Plots and Metrics

    def create_visualization(self):
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)

        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)


    # Status Bar with Progress Bar

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)


    # Helper Function to Create Parameter Groups (Algorithm Groups)

    def create_algorithm_group(self, name, params):
        """Helper to create a QGroupBox with input widgets for model parameters and a Train button."""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            # NEW: Support for text input fields (for custom Bayesian priors)
            if isinstance(param_type, tuple) and param_type[0] == "text":
                widget = QLineEdit()
                widget.setText(param_type[1])
            elif param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 10000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 10000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            else:
                widget = QLineEdit()
            widget.setObjectName(param_name)
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        group.setLayout(layout)
        return group


    # Dataset Loading with SimpleImputer for Missing Data Handling

    def load_dataset(self):
        try:
            dataset_name = self.dataset_combo.currentText()
            if dataset_name == "Load Custom Dataset":
                return

            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.load_boston()
            elif dataset_name == "MNIST Dataset":
            
                pass

            X = pd.DataFrame(data.data)
            y = data.target

          
            method = self.missing_combo.currentText()
            if method != "No Imputation":
                strategy = None
                fill_value = None
                if method == "Mean Imputation":
                    strategy = "mean"
                elif method == "Median Imputation":
                    strategy = "median"
                elif method == "Most Frequent Imputation":
                    strategy = "most_frequent"
                elif method == "Constant Imputation":
                    strategy = "constant"
                    fill_value = 0
                imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(X.values, y, test_size=test_size, random_state=42)
            self.apply_scaling()
            self.status_bar.showMessage(f"Loaded {dataset_name}")
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")


    # Custom Dataset Loading with SimpleImputer for Missing Data Handling

    def load_custom_data(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV files (*.csv)")
            if file_name:
                data = pd.read_csv(file_name)
                target_col = self.select_target_column(data.columns)
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    method = self.missing_combo.currentText()
                    if method != "No Imputation":
                        strategy = None
                        fill_value = None
                        if method == "Mean Imputation":
                            strategy = "mean"
                        elif method == "Median Imputation":
                            strategy = "median"
                        elif method == "Most Frequent Imputation":
                            strategy = "most_frequent"
                        elif method == "Constant Imputation":
                            strategy = "constant"
                            fill_value = 0
                        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X.values, y, test_size=test_size, random_state=42)
                    self.apply_scaling()
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    def select_target_column(self, columns):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        if dialog.exec():
            return combo.currentText()
        return None

    def apply_scaling(self):
        scaling_method = self.scaling_combo.currentText()
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
            except Exception as e:
                self.show_error(f"Error during scaling: {str(e)}")


    # Train Model Function - Updated for New Models, Loss Options, and Custom Bayesian Priors

    def train_model(self, name, param_widgets):
        if self.X_train is None or self.y_train is None:
            self.show_error("No dataset loaded. Please load data before training.")
            return
        try:
            model = None
            selected_loss = None

            def get_widget_value(widget):
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    return widget.value()
                elif isinstance(widget, QCheckBox):
                    return widget.isChecked()
                elif isinstance(widget, QComboBox):
                    return widget.currentText()
                elif isinstance(widget, QLineEdit):
                    return widget.text()
                else:
                    return widget

            params = {p: get_widget_value(w) for p, w in param_widgets.items()}

            # Regression Models
            if name == "Linear Regression":
                selected_loss = params.get("loss_function", None)
                fit_intercept = params.get("fit_intercept", True)
                normalize = params.get("normalize", False)
                if selected_loss == "MAE":
                    model = QuantileRegressor(quantile=0.5, alpha=0.0, max_iter=10000)
                elif selected_loss == "Huber":
                    model = HuberRegressor()
                else:
                    model = LinearRegression(fit_intercept=fit_intercept)

            elif name == "Decision Tree Regressor":
                criterion = "squared_error"
                if "loss_function" in params:
                    selected_loss = params["loss_function"]
                    if selected_loss == "MAE":
                        criterion = "absolute_error"
                    else:
                        criterion = "squared_error"
                max_depth = params.get("max_depth", None)
                min_samples_split = params.get("min_samples_split", 2)
                model = DecisionTreeRegressor(max_depth=max_depth if max_depth > 0 else None,
                                              min_samples_split=min_samples_split,
                                              criterion=criterion)

            elif name == "Support Vector Regressor":
                C = params.get("C", 1.0)
                epsilon = params.get("epsilon", 0.1)
                kernel = params.get("kernel", "rbf")
                degree = params.get("degree", 3)
                model = SVR(C=C, epsilon=epsilon, kernel=kernel, degree=degree)

            # Classification Models
            elif name == "Logistic Regression":
                selected_loss = params.get("loss_function", None)
                C = params.get("C", 1.0)
                max_iter = params.get("max_iter", 100)
                multi_class = params.get("multi_class", "ovr")
                if selected_loss == "Hinge":
                    model = LinearSVC(C=C, max_iter=max_iter, dual=True)
                else:
                    model = LogisticRegression(C=C, max_iter=max_iter, multi_class=multi_class, solver='lbfgs')

            elif name == "Decision Tree":
                max_depth = params.get("max_depth", None)
                min_samples_split = params.get("min_samples_split", 2)
                criterion = params.get("criterion", "gini")
                model = DecisionTreeClassifier(max_depth=max_depth if max_depth > 0 else None,
                                               min_samples_split=min_samples_split,
                                               criterion=criterion)

            elif name == "Random Forest":
                n_estimators = params.get("n_estimators", 10)
                max_depth = params.get("max_depth", None)
                min_samples_split = params.get("min_samples_split", 2)
                model = RandomForestClassifier(n_estimators=n_estimators,
                                               max_depth=max_depth if max_depth > 0 else None,
                                               min_samples_split=min_samples_split)

            elif name == "Support Vector Machine":
                selected_loss = params.get("loss_function", None)
                C = params.get("C", 1.0)
                kernel = params.get("kernel", "rbf")
                degree = params.get("degree", 3)
                if selected_loss == "Cross-Entropy" and kernel == "linear":
                    model = LogisticRegression(C=C, max_iter=200, solver='lbfgs')
                else:
                    model = SVC(C=C, kernel=kernel, degree=degree)

            elif name == "Naive Bayes":
                var_smoothing = params.get("var_smoothing", 1e-9)
                priors_option = params.get("priors", "Data-Derived")
                priors = None
                if priors_option == "Uniform":
                    num_classes = len(np.unique(self.y_train))
                    priors = [1.0/num_classes] * num_classes
                elif priors_option == "Custom":
                    custom_str = params.get("custom_priors", "[0.3, 0.7]")
                    try:
                        priors = ast.literal_eval(custom_str)
                    except Exception:
                        priors = None
                model = GaussianNB(var_smoothing=var_smoothing, priors=priors)

            elif name == "K-Nearest Neighbors":
                n_neighbors = params.get("n_neighbors", 5)
                weights = params.get("weights", "uniform")
                metric = params.get("metric", "euclidean")
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
            else:
                self.show_error(f"Unknown model: {name}")
                return

            self.current_model = model
            self.current_model.fit(self.X_train, self.y_train)
            if hasattr(self.current_model, "predict"):
                y_pred = self.current_model.predict(self.X_test)
            else:
                y_pred = None

            if y_pred is not None:
                self.update_visualization(y_pred)
                self.update_metrics(y_pred, selected_loss)
                self.status_bar.showMessage(f"Trained {name} model")
        except Exception as e:
            self.show_error(f"Error during training {name}: {str(e)}")


    # Visualization Panel Update (Plot and Metrics)

    def update_visualization(self, y_pred, title=None):
        self.figure.clear()
        if len(np.unique(self.y_test)) > 10:
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred, color='blue', alpha=0.6)
            min_val = min(np.min(self.y_test), np.min(y_pred))
            max_val = max(np.max(self.y_test), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Regression: Actual vs Predicted")
        else:
            ax = self.figure.add_subplot(111)
            if self.X_test.shape[1] > 2:
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
            else:
                X_test_2d = self.X_test
            scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
            ax.set_title("Classification: Test Data Predictions")
            self.figure.colorbar(scatter, ax=ax)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_metrics(self, y_pred, selected_loss=None):
        metrics_text = "Model Performance Metrics:\n\n"
        if len(np.unique(self.y_test)) > 10:
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = self.current_model.score(self.X_test, self.y_test)
            if selected_loss == "MAE":
                metrics_text += f"Mean Absolute Error: {mae:.4f}\n"
                metrics_text += f"R² Score: {r2:.4f}\n"
            elif selected_loss == "Huber":
                epsilon = getattr(self.current_model, "epsilon", 1.35)
                errors = self.y_test - y_pred
                huber_loss_vals = []
                for err in errors:
                    if abs(err) <= epsilon:
                        huber_loss_vals.append(0.5 * err**2)
                    else:
                        huber_loss_vals.append(epsilon * (abs(err) - 0.5 * epsilon))
                huber_loss_avg = np.mean(huber_loss_vals)
                metrics_text += f"Huber Loss (ε={epsilon}): {huber_loss_avg:.4f}\n"
                metrics_text += f"R² Score: {r2:.4f}\n"
            else:
                metrics_text += f"Mean Squared Error: {mse:.4f}\n"
                metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
                metrics_text += f"R² Score: {r2:.4f}\n"
        else:
            accuracy = accuracy_score(self.y_test, y_pred)
            metrics_text += f"Accuracy: {accuracy:.4f}\n"
            if selected_loss == "Cross-Entropy":
                if hasattr(self.current_model, "predict_proba"):
                    y_prob = self.current_model.predict_proba(self.X_test)
                elif hasattr(self.current_model, "decision_function"):
                    dec = self.current_model.decision_function(self.X_test)
                    if dec.ndim == 1:
                        y_prob = np.column_stack([1/(1+np.exp(dec)), 1/(1+np.exp(-dec))])
                    else:
                        exp_scores = np.exp(dec)
                        y_prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                else:
                    y_prob = None
                if y_prob is not None:
                    try:
                        ll = log_loss(self.y_test, y_prob, labels=np.unique(self.y_test))
                    except Exception:
                        ll = log_loss(self.y_test, y_prob)
                    metrics_text += f"Log Loss (Cross-Entropy): {ll:.4f}\n"
            elif selected_loss == "Hinge":
                if hasattr(self.current_model, "decision_function"):
                    dec = self.current_model.decision_function(self.X_test)
                    try:
                        hl = hinge_loss(self.y_test, dec, labels=np.unique(self.y_test))
                    except Exception:
                        hl = hinge_loss(self.y_test, dec)
                    metrics_text += f"Hinge Loss: {hl:.4f}\n"
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            metrics_text += "\nConfusion Matrix:\n"
            metrics_text += str(conf_matrix)
        self.metrics_text.setText(metrics_text)


    # Error Display Helper

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)


# Main Application Entry Point

def main():
    app = QApplication([])
    window = MLCourseGUI()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
