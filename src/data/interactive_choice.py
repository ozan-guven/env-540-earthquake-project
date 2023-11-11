import os
import shutil
import tkinter as tk
from tkinter import messagebox, Label
from PIL import Image, ImageTk
from typing import List

DATA_PATH = '../../data/'
MAXAR_FILTERED_PATCHES_PATH = DATA_PATH + 'maxar_filtered_patches/'
MAXAR_REVIEWED_PATCHES_PATH = DATA_PATH + 'maxar_reviewed_patches/'
MAXAR_ALL_PATCHES_PATH = MAXAR_REVIEWED_PATCHES_PATH + 'all/'
MAXAR_ALL_POST_PATCHES_PATH = MAXAR_ALL_PATCHES_PATH + 'post/'
MAXAR_ALL_PRE_PATCHES_PATH = MAXAR_ALL_PATCHES_PATH + 'pre/'
MAXAR_DAMAGED_PATCHES_PATH = MAXAR_REVIEWED_PATCHES_PATH + 'damaged/'
MAXAR_DAMAGED_POST_PATCHES_PATH = MAXAR_DAMAGED_PATCHES_PATH + 'post/'
MAXAR_DAMAGED_PRE_PATCHES_PATH = MAXAR_DAMAGED_PATCHES_PATH + 'pre/'
MAXAR_INTACT_PATCHES_PATH = MAXAR_REVIEWED_PATCHES_PATH + 'intact/'
MAXAR_INTACT_POST_PATCHES_PATH = MAXAR_INTACT_PATCHES_PATH + 'post/'
MAXAR_INTACT_PRE_PATCHES_PATH = MAXAR_INTACT_PATCHES_PATH + 'pre/'


def _create_data_folders() -> None:
    """Create pre and post earthquake folders for data.
    """

    # Copy filtered patches to reviewed patches if they exist
    if os.path.exists(MAXAR_FILTERED_PATCHES_PATH) and not os.path.exists(MAXAR_ALL_PATCHES_PATH):
        print('▶️ Copying filtered patches...')
        shutil.copytree(MAXAR_FILTERED_PATCHES_PATH, MAXAR_ALL_PATCHES_PATH)

    # Create data folders
    print('▶️ Creating data folders...')
    os.makedirs(MAXAR_DAMAGED_PATCHES_PATH, exist_ok=True)
    os.makedirs(MAXAR_DAMAGED_POST_PATCHES_PATH, exist_ok=True)
    os.makedirs(MAXAR_DAMAGED_PRE_PATCHES_PATH, exist_ok=True)
    os.makedirs(MAXAR_INTACT_PATCHES_PATH, exist_ok=True)
    os.makedirs(MAXAR_INTACT_POST_PATCHES_PATH, exist_ok=True)
    os.makedirs(MAXAR_INTACT_PRE_PATCHES_PATH, exist_ok=True)

class ImageReviewerApp:
    def __init__(self, root, patch_file_dict):
        self.root = root
        self.root.title("🌍 Image Reviewer")
        self.patch_file_dict = patch_file_dict

        self.current_key = None
        self.current_pre_image_file = None
        self.current_post_image_file = None

        # Configure the grid
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        # Image panels in the top row
        self.pre_panel = Label(root)
        self.pre_panel.grid(row=0, column=0, sticky="nsew")

        self.post_panel = Label(root)
        self.post_panel.grid(row=0, column=1, sticky="nsew")

        # Buttons in the bottom row
        self.keep_button = tk.Button(root, text="Damaged", command=self.keep_data)
        self.keep_button.grid(row=1, column=0, sticky="nsew")

        self.discard_button = tk.Button(root, text="Intact", command=self.discard_data)
        self.discard_button.grid(row=1, column=1, sticky="nsew")

        self.update_image()

    def update_image(self):
        # Check if there are no more images to review
        if not self.patch_file_dict:
            messagebox.showinfo("End", "No more images to review.")
            self.root.destroy()
            return

        # Get first key
        self.current_key = next(iter(self.patch_file_dict))
        if self.current_key is None:
            self.root.after(100, self.update_image)  # Schedule next update
            return

        # Check if there are no pre or post images
        if len(self.patch_file_dict[self.current_key]['pre']) < 1 or len(self.patch_file_dict[self.current_key]['post']) < 1:
            del self.patch_file_dict[self.current_key]
            self.root.after(100, self.update_image)  # Schedule next update
            return

        # Get all pre images and one post image
        self.current_pre_image_file = self.patch_file_dict[self.current_key]['pre'][0]
        self.current_post_image_file = self.patch_file_dict[self.current_key]['post'][0]

        # Show images
        def display_image(image_file, panel):
            displayed_image = Image.open(image_file)
            displayed_image = displayed_image.resize((1024, 1024), Image.ANTIALIAS)
            displayed_image = ImageTk.PhotoImage(displayed_image)
            panel.config(image=displayed_image)
            panel.image = displayed_image

        # Usage
        display_image(self.current_pre_image_file, self.pre_panel)
        display_image(self.current_post_image_file, self.post_panel)
        # Update title
        self.root.title(f"🌍 Image Reviewer ({len(self.patch_file_dict)} entries remaining)") 

        # Remove the processed key
        # del self.patch_file_dict[first_key]

    def keep_data(self):
        # Move image to damaged folder
        shutil.move(self.current_pre_image_file, MAXAR_DAMAGED_PRE_PATCHES_PATH)
        shutil.move(self.current_post_image_file, MAXAR_DAMAGED_POST_PATCHES_PATH)

        # Delete shown images
        del self.patch_file_dict[self.current_key]['pre'][0]
        del self.patch_file_dict[self.current_key]['post'][0]

        if len(self.patch_file_dict[self.current_key]['pre']) < 1 or len(self.patch_file_dict[self.current_key]['post']) < 1:
            del self.patch_file_dict[self.current_key]

        # Schedule next update
        self.root.after(100, self.update_image)

        print(f"⚠️ Damaged: {self.current_key}")

    def discard_data(self):
        # Move image to intact folder
        shutil.move(self.current_pre_image_file, MAXAR_INTACT_PRE_PATCHES_PATH)
        shutil.move(self.current_post_image_file, MAXAR_INTACT_POST_PATCHES_PATH)

        # Delete shown images
        del self.patch_file_dict[self.current_key]['pre'][0]
        del self.patch_file_dict[self.current_key]['post'][0]

        if len(self.patch_file_dict[self.current_key]['pre']) < 1 or len(self.patch_file_dict[self.current_key]['post']) < 1:
            del self.patch_file_dict[self.current_key]

        # Schedule next update
        self.root.after(100, self.update_image)

        print(f"🛡️ Intact: {self.current_key}")

def _get_short_patch_file_name(file: str) -> str:
    short_file = file.split('/')[-1]
    short_file = short_file.split('_')[0] + '_' + short_file.split('_')[3]
    short_file = short_file.split('.')[0]
    return short_file

def _get_patch_files() -> List[str]:
    dict = {}
    MAXAR_ALL_POST_PATCHES_PATH
    all_pre_patch_files = [f"{MAXAR_ALL_PRE_PATCHES_PATH}{file}" for file in os.listdir(MAXAR_ALL_PRE_PATCHES_PATH)]
    all_post_patch_files = [f"{MAXAR_ALL_POST_PATCHES_PATH}{file}" for file in os.listdir(MAXAR_ALL_POST_PATCHES_PATH)]

    # Fill dictionary with pre data
    for all_pre_patch_file in all_pre_patch_files:
        key = _get_short_patch_file_name(all_pre_patch_file)
        if key not in dict:
            dict[key] = {}
            dict[key]['pre'] = []
            dict[key]['post'] = []

        dict[key]['pre'].append(all_pre_patch_file)

    # Fill dictionary with post data
    for all_post_patch_file in all_post_patch_files:
        key = _get_short_patch_file_name(all_post_patch_file)
        if key not in dict:
            dict[key] = {}
            dict[key]['pre'] = []
            dict[key]['post'] = []

        dict[key]['post'].append(all_post_patch_file)

    # Remove entries with no pre or no post data
    for key in list(dict):
        if len(dict[key]['pre']) < 1 or len(dict[key]['post']) < 1:
            del dict[key]

    return dict


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("2148x1124+300+150")
    root.resizable(width=True, height=True)

    _create_data_folders()
    patch_file_dict = _get_patch_files()

    app = ImageReviewerApp(root, patch_file_dict)
    root.mainloop()