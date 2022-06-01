from manim import *


class Example(ZoomedScene):
    def __init__(self, **kwargs):  # HEREFROM
        ZoomedScene.__init__(
            self,
            zoom_factor=0.1,
            zoomed_display_height=6,
            zoomed_display_width=3,
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
            },
            **kwargs
        )

    def construct(self):
        self.activate_zooming(animate=False)

        ax = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=2,
            y_length=2,
            x_axis_config={"color": ORANGE},
            y_axis_config={"color": ORANGE},
        )
        ax.shift(DL)
        x_vals = [0, 1, 2, 3, 4, 5]
        y_vals = [2, -1, 4, 2, 4, 1]
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals)
        self.zoomed_camera.frame.move_to(graph.get_top() + 0.1 * DL)
        self.zoomed_display.shift(3 * LEFT + 0.4 * UP)
        self.camera.frame.scale(1 / 2)
        self.camera.frame.shift(UR * 1)
        self.add(ax, graph)  # HERETO
