# src/classic_ml/app.py

import os
import json
import pandas as pd
from pathlib import Path
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Static, Input, Label, Select, Button, ContentSwitcher, Footer
from textual.reactive import reactive
from textual_plotext import PlotextPlot

# Import local modules
from . import i18n
from .algorithms.kmeans import KMeans
from .algorithms.linear_regression import LinearRegression
from .algorithms.perceptron import Perceptron

PALETTE = {
    "bg": "#000000",
    "primary": "#00f5d4",  # Cyan
    "text": "#ffffff",    # White
}

class ClassicMLApp(App):
    CSS = f"""
    Screen {{
        background: {PALETTE["bg"]};
        color: {PALETTE["text"]};
    }}
    
    #sidebar {{
        width: 42;
        height: 100%;
        padding: 1 2;
        background: #050505;
        border: round {PALETTE["primary"]};
        border-title-align: left;
        margin: 1;
    }}

    #sidebar_scroll {{
        width: 100%;
        height: 100%;
        overflow-y: auto;
    }}
    
    #logo {{
        width: 100%;
        height: 10;
        content-align: center middle;
        text-align: center;
        background: black;
        color: {PALETTE["primary"]};
        padding: 2 0 2 0;
        margin: 0;
    }}

    #main_content {{
        width: 1fr;
        height: 100%;
    }}
    
    #results_area {{
        height: 12;
        padding: 1 2;
        background: #0a0a0a;
        margin: 1 1 0 0;
        border: round {PALETTE["primary"]};
        border-title-align: left;
    }}

    #plot {{
        height: 1fr;
        background: #0a0a0a;
        margin: 0 1 1 0;
        border: round {PALETTE["primary"]};
        border-title-align: left;
    }}
    
    Input, Select, Button {{
        background: #111111;
        border: solid {PALETTE["primary"]};
        color: {PALETTE["text"]};
        height: 3;
        margin-bottom: 1;
    }}

    #btn_algo_toggle, #btn_mode_toggle {{
        width: 100%;
        height: 3;
        border: solid #333333;
        background: #1a1a1a;
        color: white;
        text-style: bold;
        padding: 0 2;
        margin: 0;
        text-align: center;
    }}

    #btn_algo_toggle:focus, #btn_mode_toggle:focus {{
        border: solid {PALETTE["primary"]};
    }}

    #algo_menu, #mode_menu {{
        display: none;
        width: 100%;
        height: auto;
        border: solid #333333;
        background: #050505;
        margin-top: 0;
    }}

    #algo_menu Button, #mode_menu Button {{
        width: 100%;
        height: 1;
        border: none;
        background: transparent;
        color: white;
        text-align: left;
        padding: 0 2;
        margin: 0;
    }}

    #algo_menu Button:hover, #mode_menu Button:hover {{
        background: #111111;
        color: {PALETTE["primary"]};
    }}

    #results_text {{
        color: {PALETTE["text"]};
        padding: 0 1;
    }}
    
    ContentSwitcher {{
        height: auto;
    }}

    /* Prevent Vertical containers inside the sidebar from taking up 1fr height */
    ContentSwitcher > Vertical, #perc_init_container, #perc_grid_container {{
        height: auto;
    }}

    
    Input:focus, Select:focus, Button:focus {{
        background: #222222;
        border: double {PALETTE["primary"]};
    }}
    
    Label {{
        color: {PALETTE["primary"]};
        text-style: bold;
        margin-top: 1;
    }}
    """
    
    # Reactive state
    current_algo = reactive("kmeans")
    csv_path = reactive("")
    k_value = reactive(3)
    max_iter = reactive(100)
    learning_rate = reactive(0.1)
    epochs = reactive(100)
    reg_lambda = reactive(0.0)
    
    # New reactive state
    perceptron_mode = reactive("binary")
    predict_val = reactive("")
    initial_weights = reactive("") # Comma separated
    manual_bias = reactive(0.0)
    grid_range_w0 = reactive("-1,1,0.1")
    grid_range_w1 = reactive("-1,1,0.1")

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("l", "toggle_lang", "Language"),
        ("i", "load_iris", "Load Iris"),
        ("r", "reset", "Reset"),
    ]

    def compose(self) -> ComposeResult:
        logo_path = Path(__file__).parent / "data" / "logo.txt"
        logo_text = ""
        if logo_path.exists():
            logo_text = logo_path.read_text(encoding="utf-8")
        
        yield Static(logo_text, id="logo")
        
        yield Horizontal(
            Vertical(
                ScrollableContainer(
                    Label(i18n.t("select_algo"), id="lbl_select_algo"),
                    Button(f"▼ {i18n.t(f'menu_{self.current_algo}')}", id="btn_algo_toggle"),
                    Vertical(
                        Button(i18n.t("menu_kmeans"), id="btn_sel_kmeans"),
                        Button(i18n.t("menu_linear"), id="btn_sel_linear"),
                        Button(i18n.t("menu_perceptron"), id="btn_sel_perceptron"),
                        id="algo_menu"
                    ),
                    Label(i18n.t("label_path"), id="lbl_path"),
                    Input(placeholder="data/iris.csv", id="inp_path"),
                    
                    ContentSwitcher(
                        Vertical(
                            Label(i18n.t("label_k"), id="lbl_k"),
                            Input(str(self.k_value), id="inp_k"),
                            Label(i18n.t("label_iters"), id="lbl_iters"),
                            Input(str(self.max_iter), id="inp_iters"),
                            id="kmeans_inputs"
                        ),
                        Vertical(
                            Label(i18n.t("label_lambda"), id="lbl_lambda"),
                            Input(str(self.reg_lambda), id="inp_lambda"),
                            id="linear_inputs"
                        ),
                        Vertical(
                            Label(i18n.t("label_mode"), id="lbl_mode"),
                            Button(f"▼ {i18n.t(f'mode_{self.perceptron_mode}')}", id="btn_mode_toggle"),
                            Vertical(
                                Button(i18n.t("mode_binary"), id="btn_mode_binary"),
                                Button(i18n.t("mode_ovr"), id="btn_mode_ovr"),
                                Button(i18n.t("mode_test"), id="btn_mode_test"),
                                Button(i18n.t("mode_grid"), id="btn_mode_grid"),
                                id="mode_menu"
                            ),
                            Label(i18n.t("label_lr"), id="lbl_lr"),
                            Input(str(self.learning_rate), id="inp_lr"),
                            Label(i18n.t("label_epochs"), id="lbl_epochs"),
                            Input(str(self.epochs), id="inp_epochs"),
                            
                            Vertical(
                                Label(i18n.t("label_w_init"), id="lbl_w_init"),
                                Input(placeholder="0.0, 0.0", id="inp_w_init"),
                                id="perc_init_container"
                            ),
                            Vertical(
                                Label(i18n.t("label_bias"), id="lbl_bias"),
                                Input(str(self.manual_bias), id="inp_bias"),
                                Label(i18n.t("label_range") + " W0", id="lbl_range_w0"),
                                Input(self.grid_range_w0, id="inp_range_w0"),
                                Label(i18n.t("label_range") + " W1", id="lbl_range_w1"),
                                Input(self.grid_range_w1, id="inp_range_w1"),
                                id="perc_grid_container"
                            ),
                            id="perceptron_inputs"
                        ),
                        id="algo_switcher",
                        initial="kmeans_inputs"
                    ),
                    Label(i18n.t("label_predict"), id="lbl_predict"),
                    Input(placeholder="5.1, 3.5, 1.4, 0.2", id="inp_predict"),
                    Button(i18n.t("btn_run"), id="btn_run", variant="success"),
                    id="sidebar_scroll"
                ),
                id="sidebar"
            ),
            Vertical(
                ScrollableContainer(
                    Static(id="results_text"),
                    id="results_area"
                ),
                PlotextPlot(id="plot"),
                id="main_content"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = i18n.t("app_title")
        self.query_one("#logo").styles.text_align = "center"
        
        self.query_one("#sidebar").border_title = i18n.t("title_config")
        self.query_one("#results_area").border_title = i18n.t("title_results")
        self.query_one("#plot").border_title = i18n.t("title_viz")

        self.check_logo_visibility()

        self.csv_path = "iris.csv"
        self.action_load_iris()
        self.update_inputs_view()
        self.update_perceptron_extras()

    def update_perceptron_extras(self):
        try:
            init_cont = self.query_one("#perc_init_container")
            grid_cont = self.query_one("#perc_grid_container")
            
            init_cont.display = self.perceptron_mode in ["binary", "ovr", "test"]
            grid_cont.display = self.perceptron_mode == "grid"
            
            bias_label = self.query_one("#lbl_bias")
            if self.perceptron_mode == "grid":
                bias_label.update(i18n.t("label_bias"))
            else:
                bias_label.update(i18n.t("label_bias") + " (initial)")
        except:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_run":
            self.run_algorithm()
        elif event.button.id == "btn_algo_toggle":
            menu = self.query_one("#algo_menu", Vertical)
            menu.display = not menu.display
        elif event.button.id == "btn_mode_toggle":
            menu = self.query_one("#mode_menu", Vertical)
            menu.display = not menu.display
        elif event.button.id.startswith("btn_sel_"):
            selection = event.button.id.replace("btn_sel_", "")
            self.current_algo = selection
            self.query_one("#btn_algo_toggle", Button).label = f"▼ {i18n.t(f'menu_{self.current_algo}')}"
            self.query_one("#algo_menu", Vertical).display = False
            self.update_inputs_view()
            self.run_algorithm()
        elif event.button.id.startswith("btn_mode_"):
            selection = event.button.id.replace("btn_mode_", "")
            self.perceptron_mode = selection
            self.query_one("#btn_mode_toggle", Button).label = f"▼ {i18n.t(f'mode_{self.perceptron_mode}')}"
            self.query_one("#mode_menu", Vertical).display = False
            self.update_perceptron_extras()
            self.run_algorithm()

    def on_resize(self, event: events.Resize) -> None:
        self.check_logo_visibility()

    def check_logo_visibility(self) -> None:
        logo = self.query_one("#logo")
        if self.size.height < 35:
            logo.display = False
        else:
            logo.display = True

    def update_inputs_view(self):
        switcher = self.query_one("#algo_switcher", ContentSwitcher)
        if self.current_algo == "kmeans":
            switcher.current = "kmeans_inputs"
        elif self.current_algo == "linear":
            switcher.current = "linear_inputs"
        elif self.current_algo == "perceptron":
            switcher.current = "perceptron_inputs"

    def action_toggle_lang(self):
        new_lang = i18n.toggle_language()
        self.title = i18n.t("app_title")
        self.query_one("#lbl_select_algo", Label).update(i18n.t("select_algo"))
        self.query_one("#lbl_path", Label).update(i18n.t("label_path"))
        self.query_one("#btn_run", Button).label = i18n.t("btn_run")
        
        self.query_one("#lbl_k", Label).update(i18n.t("label_k"))
        self.query_one("#lbl_iters", Label).update(i18n.t("label_iters"))
        self.query_one("#lbl_lambda", Label).update(i18n.t("label_lambda"))
        self.query_one("#lbl_lr", Label).update(i18n.t("label_lr"))
        self.query_one("#lbl_epochs", Label).update(i18n.t("label_epochs"))
        self.query_one("#lbl_mode", Label).update(i18n.t("label_mode"))
        self.query_one("#lbl_w_init", Label).update(i18n.t("label_w_init"))
        self.query_one("#lbl_bias", Label).update(i18n.t("label_bias"))
        self.query_one("#lbl_predict", Label).update(i18n.t("label_predict"))
        
        self.query_one("#btn_algo_toggle", Button).label = f"▼ {i18n.t(f'menu_{self.current_algo}')}"
        self.query_one("#btn_sel_kmeans", Button).label = i18n.t("menu_kmeans")
        self.query_one("#btn_sel_linear", Button).label = i18n.t("menu_linear")
        self.query_one("#btn_sel_perceptron", Button).label = i18n.t("menu_perceptron")

        self.query_one("#btn_mode_toggle", Button).label = f"▼ {i18n.t(f'mode_{self.perceptron_mode}')}"
        self.query_one("#btn_mode_binary", Button).label = i18n.t("mode_binary")
        self.query_one("#btn_mode_ovr", Button).label = i18n.t("mode_ovr")
        self.query_one("#btn_mode_test", Button).label = i18n.t("mode_test")
        self.query_one("#btn_mode_grid", Button).label = i18n.t("mode_grid")

        self.query_one("#sidebar").border_title = i18n.t("title_config")
        self.query_one("#results_area").border_title = i18n.t("title_results")
        self.query_one("#plot").border_title = i18n.t("title_viz")

        self.run_algorithm()
        self.notify(f"Language set to: {new_lang.upper()}")

    def action_load_iris(self):
        iris_path = str(Path(__file__).parent / "data" / "iris.csv")
        self.csv_path = iris_path
        self.query_one("#inp_path", Input).value = "iris.csv"
        self.run_algorithm()

    def on_input_changed(self, event: Input.Changed) -> None:
        try:
            val = event.value
            if event.input.id == "inp_path":
                if val == "iris.csv":
                    self.csv_path = str(Path(__file__).parent / "data" / "iris.csv")
                else:
                    self.csv_path = val
            elif event.input.id == "inp_k":
                self.k_value = int(val)
            elif event.input.id == "inp_iters":
                self.max_iter = int(val)
            elif event.input.id == "inp_lr":
                self.learning_rate = float(val)
            elif event.input.id == "inp_epochs":
                self.epochs = int(val)
            elif event.input.id == "inp_lambda":
                self.reg_lambda = float(val)
            elif event.input.id == "inp_predict":
                self.predict_val = val
            elif event.input.id == "inp_w_init":
                self.initial_weights = val
            elif event.input.id == "inp_bias":
                self.manual_bias = float(val)
            elif event.input.id == "inp_range_w0":
                self.grid_range_w0 = val
            elif event.input.id == "inp_range_w1":
                self.grid_range_w1 = val
            
            self.run_algorithm()
        except ValueError:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.run_algorithm()

    def run_algorithm(self):
        if not self.csv_path or not os.path.exists(self.csv_path):
            return
        
        try:
            df = pd.read_csv(self.csv_path)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols: return
            
            X = df[numeric_cols].values.tolist()
            
            def pretty_format(obj, indent=2):
                def round_data(o):
                    if isinstance(o, list): return [round_data(x) for x in o]
                    if isinstance(o, float): return round(o, 4)
                    return o
                return str(round_data(obj))

            plt = self.query_one("#plot").plt
            plt.clf()
            plt.theme("textual-design-dark")
            
            res_text = ""
            
            if self.current_algo == "kmeans":
                model = KMeans(k=self.k_value, max_iter=self.max_iter)
                centroids, clusters = model.fit(X)
                if len(numeric_cols) >= 2:
                    for i, cluster in enumerate(clusters):
                        if cluster:
                            plt.scatter([p[0] for p in cluster], [p[1] for p in cluster], label=f"C{i}")
                    plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], marker="f", label="Centroids")
                res_text = i18n.t("msg_converged", i=model.iterations) + f"\n\nCentroids:\n{pretty_format(centroids)}"

            elif self.current_algo == "linear":
                if len(numeric_cols) < 2: return
                X_lr = df[numeric_cols[:-1]].values.tolist()
                y_lr = df[numeric_cols[-1]].values.tolist()
                model = LinearRegression(lam=self.reg_lambda)
                weights = model.fit(X_lr, y_lr)
                if len(numeric_cols) == 2:
                    xs = [row[0] for row in X_lr]
                    plt.scatter(xs, y_lr, label="Data")
                    x_line = [min(xs), max(xs)]
                    plt.plot(x_line, [model.predict([x]) for x in x_line], color="red")
                res_text = f"Weights:\n{pretty_format(weights)}"

            elif self.current_algo == "perceptron":
                y_all = df.iloc[:, -1].values.tolist()
                classes = sorted(list(set(y_all)))
                X_p = df[numeric_cols[:2]].values.tolist()
                
                # Manual Weights
                init_w = [float(v) for v in self.initial_weights.split(",")] if self.initial_weights else None

                if self.perceptron_mode == "binary":
                    pos_class = classes[0]
                    y_bin = [1 if y == pos_class else -1 for y in y_all]
                    model = Perceptron(learning_rate=self.learning_rate, epochs=self.epochs)
                    w, b, _ = model.fit(X_p, y_bin, init_w=init_w, init_b=self.manual_bias)
                    
                    pos_points = [x for x, y in zip(X_p, y_bin) if y == 1]
                    neg_points = [x for x, y in zip(X_p, y_bin) if y == -1]
                    plt.scatter([p[0] for p in pos_points], [p[1] for p in pos_points], label=str(pos_class))
                    plt.scatter([p[0] for p in neg_points], [p[1] for p in neg_points], label="Other")
                    
                    if len(w) == 2 and w[1] != 0:
                        x_min, x_max = min(p[0] for p in X_p), max(p[0] for p in X_p)
                        plt.plot([x_min, x_max], [-(w[0]*x_min + b)/w[1], -(w[0]*x_max + b)/w[1]], color="yellow")
                    res_text = i18n.t("msg_accuracy", acc=round(model.evaluate(X_p, y_bin), 2)) + f"\nw: {pretty_format(w)}\nb: {round(b,4)}"
                
                elif self.perceptron_mode == "ovr":
                    classifiers = Perceptron.train_ovr(X_p, y_all, lr=self.learning_rate, epochs=self.epochs)
                    for cls in classes:
                        pts = [x for x, y in zip(X_p, y_all) if y == cls]
                        plt.scatter([p[0] for p in pts], [p[1] for p in pts], label=str(cls))
                    res_text = "OvR Classifiers trained."
                    
                elif self.perceptron_mode == "grid":
                    w0r = tuple(map(float, self.grid_range_w0.split(",")))
                    w1r = tuple(map(float, self.grid_range_w1.split(",")))
                    pos_class = classes[0]
                    y_bin = [1 if y == pos_class else -1 for y in y_all]
                    solutions = Perceptron.grid_search(X_p, y_bin, self.manual_bias, w0r, w1r)
                    res_text = i18n.t("msg_grid_solutions", n=len(solutions))
                    for cls in classes:
                        pts = [x for x, y in zip(X_p, y_all) if y == cls]
                        plt.scatter([p[0] for p in pts], [p[1] for p in pts], label=str(cls))
                
                elif self.perceptron_mode == "test":
                    if init_w and len(init_w) == 2:
                        model = Perceptron()
                        model.weights, model.bias = init_w, self.manual_bias
                        pos_class = classes[0]
                        y_bin = [1 if y == pos_class else -1 for y in y_all]
                        acc = model.evaluate(X_p, y_bin)
                        res_text = i18n.t("msg_accuracy", acc=round(acc, 2))
                        x_min, x_max = min(p[0] for p in X_p), max(p[0] for p in X_p)
                        plt.plot([x_min, x_max], [-(init_w[0]*x_min + model.bias)/init_w[1], -(init_w[0]*x_max + model.bias)/init_w[1]], color="yellow")
                    else:
                        res_text = "Input 2 weights."
                    for cls in classes:
                        pts = [x for x, y in zip(X_p, y_all) if y == cls]
                        plt.scatter([p[0] for p in pts], [p[1] for p in pts], label=str(cls))

            self.query_one("#results_text").update(res_text)

            if self.predict_val:
                try:
                    p_point = [float(v) for v in self.predict_val.split(",")]
                    if self.current_algo == "perceptron" and self.perceptron_mode == "ovr":
                        pred = Perceptron.predict_ovr(classifiers, p_point[:2])
                    elif self.current_algo == "perceptron":
                        idx = model.predict(p_point[:2])[1]
                        pred = classes[0] if idx == 1 else "Other"
                    elif self.current_algo == "kmeans":
                        pred = f"Cluster {model.predict(p_point)}"
                    else:
                        pred = round(model.predict(p_point), 4)
                    self.query_one("#results_text").update(res_text + f"\n\n[bold cyan]" + i18n.t("msg_pred_result", res=pred) + "[/]")
                except: pass

            plt.title(i18n.t(f"{self.current_algo}_title"))
            self.query_one("#plot").refresh()

        except Exception as e:
            self.query_one("#results_text").update(i18n.t("msg_error_load", e=str(e)))

    def action_reset(self):
        self.k_value = 3
        self.learning_rate = 0.1
        self.epochs = 100
        self.reg_lambda = 0.0
        self.run_algorithm()

def main():
    ClassicMLApp().run()

if __name__ == "__main__":
    main()
