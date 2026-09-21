from typing import Any
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined


class Renderer:
    def __init__(self, template_dir: Path) -> None:
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
        )
        print("Renderer initialized.")

    def render(self, template_name: str, **context: Any) -> str:
        template = self.env.get_template(template_name)

        rendered_text = template.render(**context)
        return rendered_text

    def render_to_file(
        self,
        output_file: Path,
        template_name: str,
        **context: Any,
    ) -> None:
        rendered_text = self.render(template_name, **context)
        output_file.write_text(rendered_text)
        print(f"{output_file} generated successfully!")
