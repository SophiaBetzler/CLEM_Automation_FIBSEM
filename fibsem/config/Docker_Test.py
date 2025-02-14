# test_gui.py
import tkinter as tk


def main():
    root = tk.Tk()
    root.title("Docker GUI Test")
    root.geometry("300x100")

    label = tk.Label(root, text="Hello from Docker GUI!")
    label.pack(pady=10)

    close_button = tk.Button(root, text="Close", command=root.destroy)
    close_button.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
