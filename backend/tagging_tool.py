from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from backend.tagging_store import DEFAULT_PRESETS, ImageTagStore


PREVIEW_SIZE = (900, 700)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review images in a folder and assign JSONL tags with a small tkinter UI."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Root directory to scan recursively for images. If omitted, a folder picker opens.",
    )
    parser.add_argument(
        "--tags-file",
        type=Path,
        default=None,
        help="JSONL file to load/save tags. Defaults to <image-root>/image_tags.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional image limit for quick testing.",
    )
    return parser.parse_args()


def choose_directory() -> Path | None:
    picker = tk.Tk()
    picker.withdraw()
    selected = filedialog.askdirectory(title="Choose image root")
    picker.destroy()
    if not selected:
        return None
    return Path(selected)


class TaggingToolApp(tk.Tk):
    def __init__(self, store: ImageTagStore, *, limit: int | None = None) -> None:
        super().__init__()
        self.store = store
        self.limit = limit
        self.title("Sneaker Tagging Tool")
        self.geometry("1600x980")
        self.minsize(1200, 800)

        self.search_var = tk.StringVar()
        self.only_untagged_var = tk.BooleanVar(value=False)
        self.preset_var = tk.StringVar(value=next(iter(DEFAULT_PRESETS), ""))
        self.custom_tag_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.image_info_var = tk.StringVar()
        self.current_tags_var = tk.StringVar()

        self.tag_vars: dict[str, tk.BooleanVar] = {}
        self.tag_buttons: dict[str, ttk.Checkbutton] = {}
        self.display_images: list[str] = []
        self.current_index = -1
        self.current_photo: ImageTk.PhotoImage | None = None
        self.updating_tag_state = False
        self.previous_image: str | None = None

        self._build_layout()
        self._bind_shortcuts()
        self._rebuild_tag_controls()
        self.refresh_image_list(reset_selection=True)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0, minsize=340)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0, minsize=340)
        self.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(self, padding=12)
        top_bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        top_bar.columnconfigure(8, weight=1)

        ttk.Label(top_bar, text="Search").grid(row=0, column=0, sticky="w")
        search_entry = ttk.Entry(top_bar, textvariable=self.search_var, width=40)
        search_entry.grid(row=0, column=1, padx=(8, 12), sticky="ew")
        search_entry.bind("<KeyRelease>", self._on_filter_changed)

        ttk.Checkbutton(
            top_bar,
            text="Only untagged",
            variable=self.only_untagged_var,
            command=lambda: self.refresh_image_list(reset_selection=True),
        ).grid(row=0, column=2, padx=(0, 12), sticky="w")

        ttk.Button(top_bar, text="Save", command=self.save_tags).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(top_bar, text="Prev", command=self.show_previous_image).grid(
            row=0, column=4, padx=(0, 8)
        )
        ttk.Button(top_bar, text="Next", command=self.show_next_image).grid(row=0, column=5, padx=(0, 8))
        ttk.Button(top_bar, text="Delete Image", command=self.delete_current_image).grid(
            row=0,
            column=6,
            padx=(0, 8),
        )
        ttk.Button(top_bar, text="Copy Prev Tags", command=self.copy_previous_tags).grid(
            row=0,
            column=7,
            padx=(0, 12),
        )

        ttk.Label(top_bar, textvariable=self.status_var).grid(row=0, column=8, sticky="e")

        left_panel = ttk.Frame(self, padding=(12, 0, 8, 12))
        left_panel.grid(row=1, column=0, sticky="nsew")
        left_panel.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)

        ttk.Label(left_panel, text="Images").grid(row=0, column=0, sticky="w", pady=(0, 8))

        list_frame = ttk.Frame(left_panel)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.image_listbox = tk.Listbox(list_frame, exportselection=False)
        self.image_listbox.grid(row=0, column=0, sticky="nsew")
        self.image_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        list_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.image_listbox.yview)
        list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.image_listbox.configure(yscrollcommand=list_scrollbar.set)

        center_panel = ttk.Frame(self, padding=(8, 0, 8, 12))
        center_panel.grid(row=1, column=1, sticky="nsew")
        center_panel.rowconfigure(1, weight=1)
        center_panel.columnconfigure(0, weight=1)

        ttk.Label(center_panel, textvariable=self.image_info_var).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.preview_label = ttk.Label(center_panel, anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew")

        ttk.Label(
            center_panel,
            textvariable=self.current_tags_var,
            wraplength=860,
            justify="left",
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))

        right_panel = ttk.Frame(self, padding=(8, 0, 12, 12))
        right_panel.grid(row=1, column=2, sticky="nsew")
        right_panel.rowconfigure(4, weight=1)
        right_panel.columnconfigure(0, weight=1)

        ttk.Label(right_panel, text="Presets").grid(row=0, column=0, sticky="w")

        preset_row = ttk.Frame(right_panel)
        preset_row.grid(row=1, column=0, sticky="ew", pady=(6, 12))
        preset_row.columnconfigure(0, weight=1)

        preset_names = list(self.store.presets)
        self.preset_combo = ttk.Combobox(
            preset_row,
            textvariable=self.preset_var,
            values=preset_names,
            state="readonly",
        )
        self.preset_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(preset_row, text="Apply", command=self.apply_selected_preset).grid(
            row=0,
            column=1,
            padx=(8, 0),
        )

        custom_row = ttk.Frame(right_panel)
        custom_row.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        custom_row.columnconfigure(0, weight=1)

        ttk.Label(right_panel, text="Add Custom Tag").grid(row=2, column=0, sticky="nw")
        custom_entry = ttk.Entry(custom_row, textvariable=self.custom_tag_var)
        custom_entry.grid(row=0, column=0, sticky="ew", pady=(24, 0))
        custom_entry.bind("<Return>", lambda _event: self.add_custom_tag())
        ttk.Button(custom_row, text="Add", command=self.add_custom_tag).grid(
            row=0,
            column=1,
            padx=(8, 0),
            pady=(24, 0),
        )

        ttk.Label(right_panel, text="Tags").grid(row=3, column=0, sticky="w", pady=(4, 8))

        self.tags_frame = ttk.Frame(right_panel)
        self.tags_frame.grid(row=4, column=0, sticky="nsew")
        self.tags_frame.columnconfigure(0, weight=1)

        hint_lines = [
            "Shortcuts:",
            "Left/Right = prev/next image",
            "1-9 = toggle first tags in the list",
            "Ctrl+S = save",
            "Delete = remove current image",
            "U = toggle untagged filter",
        ]
        ttk.Label(
            right_panel,
            text="\n".join(hint_lines),
            justify="left",
        ).grid(row=5, column=0, sticky="sw", pady=(12, 0))

    def _bind_shortcuts(self) -> None:
        self.bind("<Left>", lambda _event: self.show_previous_image())
        self.bind("<Right>", lambda _event: self.show_next_image())
        self.bind("<Control-s>", lambda _event: self.save_tags())
        self.bind("<u>", lambda _event: self.toggle_only_untagged())
        self.bind("<U>", lambda _event: self.toggle_only_untagged())
        self.bind("<Delete>", lambda _event: self.delete_current_image())

        for index in range(1, 10):
            self.bind(str(index), lambda _event, idx=index - 1: self.toggle_tag_by_index(idx))

    def _rebuild_tag_controls(self) -> None:
        for child in self.tags_frame.winfo_children():
            child.destroy()

        self.tag_vars = {}
        self.tag_buttons = {}

        for row_index, tag in enumerate(self.store.known_tags()):
            var = tk.BooleanVar(value=False)
            checkbutton = ttk.Checkbutton(
                self.tags_frame,
                text=tag,
                variable=var,
                command=lambda current_tag=tag: self.on_tag_toggle(current_tag),
            )
            checkbutton.grid(row=row_index, column=0, sticky="w", pady=2)
            self.tag_vars[tag] = var
            self.tag_buttons[tag] = checkbutton

    def _on_filter_changed(self, _event: tk.Event[tk.Entry]) -> None:
        self.refresh_image_list(reset_selection=True)

    def refresh_image_list(self, *, reset_selection: bool) -> None:
        self.display_images = self.store.filtered_images(
            query=self.search_var.get(),
            show_only_untagged=self.only_untagged_var.get(),
        )
        if self.limit is not None:
            self.display_images = self.display_images[: self.limit]

        self.image_listbox.delete(0, tk.END)
        for image in self.display_images:
            tags = self.store.tags_for(image)
            marker = "●" if tags else "○"
            self.image_listbox.insert(tk.END, f"{marker} {image}")

        total = len(self.store.image_paths)
        visible = len(self.display_images)
        tagged = sum(1 for image in self.store.image_paths if self.store.tags_for(image))
        self.status_var.set(f"Visible {visible}/{total} | Tagged {tagged}")

        if not self.display_images:
            self.current_index = -1
            self.previous_image = None
            self.preview_label.configure(image="", text="No images match the current filter.")
            self.image_info_var.set("No image selected")
            self.current_tags_var.set("")
            return

        if reset_selection or self.current_index < 0 or self.current_index >= len(self.display_images):
            self.select_image_index(0)
            return

        self.select_image_index(self.current_index)

    def select_image_index(self, index: int) -> None:
        if not self.display_images:
            return

        bounded_index = max(0, min(index, len(self.display_images) - 1))
        self.previous_image = self.current_image_path()
        self.current_index = bounded_index
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(bounded_index)
        self.image_listbox.see(bounded_index)
        self.image_listbox.activate(bounded_index)
        self.show_image(self.display_images[bounded_index])

    def _on_listbox_select(self, _event: tk.Event[tk.Listbox]) -> None:
        selection = self.image_listbox.curselection()
        if not selection:
            return
        self.select_image_index(int(selection[0]))

    def show_image(self, relative_image_path: str) -> None:
        image_path = self.store.image_root / relative_image_path

        with Image.open(image_path) as image:
            preview = image.convert("RGB")
            preview.thumbnail(PREVIEW_SIZE)
            self.current_photo = ImageTk.PhotoImage(preview)

        self.preview_label.configure(image=self.current_photo, text="")
        image_number = self.current_index + 1
        total = len(self.display_images)
        self.image_info_var.set(f"{image_number}/{total}  {relative_image_path}")
        self._sync_tag_controls(relative_image_path)

    def _sync_tag_controls(self, relative_image_path: str) -> None:
        image_tags = self.store.tag_set_for(relative_image_path)
        self.updating_tag_state = True
        try:
            for tag, var in self.tag_vars.items():
                var.set(tag in image_tags)
        finally:
            self.updating_tag_state = False

        if image_tags:
            self.current_tags_var.set("Current tags: " + ", ".join(sorted(image_tags)))
        else:
            self.current_tags_var.set("Current tags: none")

    def current_image_path(self) -> str | None:
        if self.current_index < 0 or self.current_index >= len(self.display_images):
            return None
        return self.display_images[self.current_index]

    def on_tag_toggle(self, tag: str) -> None:
        if self.updating_tag_state:
            return

        image = self.current_image_path()
        if image is None:
            return

        enabled = self.tag_vars[tag].get()
        self.store.toggle_tag(image, tag, enabled)
        self.store.save()
        self._sync_tag_controls(image)
        self.refresh_image_list(reset_selection=False)

    def toggle_tag_by_index(self, index: int) -> None:
        tags = self.store.known_tags()
        if index < 0 or index >= len(tags):
            return
        tag = tags[index]
        current_value = self.tag_vars[tag].get()
        self.tag_vars[tag].set(not current_value)
        self.on_tag_toggle(tag)

    def apply_selected_preset(self) -> None:
        image = self.current_image_path()
        if image is None:
            return

        preset_name = self.preset_var.get().strip()
        if not preset_name:
            return

        self.store.apply_preset(image, preset_name)
        self.store.save()
        self._sync_tag_controls(image)
        self.refresh_image_list(reset_selection=False)

    def add_custom_tag(self) -> None:
        image = self.current_image_path()
        if image is None:
            return

        custom_tag = self.custom_tag_var.get().strip()
        if not custom_tag:
            return

        self.store.add_tags(image, [custom_tag])
        self.custom_tag_var.set("")
        self.store.save()
        self._rebuild_tag_controls()
        self._sync_tag_controls(image)
        self.refresh_image_list(reset_selection=False)

    def copy_previous_tags(self) -> None:
        image = self.current_image_path()
        previous_image = self.previous_image
        if image is None or previous_image is None or image == previous_image:
            return

        self.store.set_tags(image, self.store.tag_set_for(previous_image))
        self.store.save()
        self._sync_tag_controls(image)
        self.refresh_image_list(reset_selection=False)

    def toggle_only_untagged(self) -> None:
        self.only_untagged_var.set(not self.only_untagged_var.get())
        self.refresh_image_list(reset_selection=True)

    def delete_current_image(self) -> None:
        image = self.current_image_path()
        if image is None:
            return

        confirmed = messagebox.askyesno(
            "Delete image",
            f"Delete this image permanently?\n\n{image}\n\nThis cannot be undone.",
            icon="warning",
        )
        if not confirmed:
            return

        current_index = self.current_index
        self.store.delete_image(image)
        self.store.save()
        self.refresh_image_list(reset_selection=True)

        if self.display_images:
            self.select_image_index(min(current_index, len(self.display_images) - 1))

    def show_previous_image(self) -> None:
        if not self.display_images:
            return
        self.select_image_index(self.current_index - 1)

    def show_next_image(self) -> None:
        if not self.display_images:
            return
        self.select_image_index(self.current_index + 1)

    def save_tags(self) -> None:
        self.store.save()
        messagebox.showinfo("Saved", f"Tags saved to:\n{self.store.tags_file}")


def main() -> None:
    args = parse_args()
    image_root = args.image_root
    if image_root is None:
        image_root = choose_directory()
        if image_root is None:
            raise SystemExit("No image directory selected.")

    store = ImageTagStore(
        image_root=image_root,
        tags_file=args.tags_file,
    )
    app = TaggingToolApp(store, limit=args.limit)
    app.mainloop()


if __name__ == "__main__":
    main()
