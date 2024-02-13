import phonenumbers
from phonenumbers import timezone, geocoder, carrier, NumberParseException
import csv
import os
from tkinter import Tk, Label, Entry, Button, Text, Scrollbar, VERTICAL, messagebox, filedialog, Menu

def parse_phone_number(phone_number):
    """Parse and validate a phone number, returning detailed information if valid."""
    try:
        parsed_number = phonenumbers.parse(phone_number)
        if phonenumbers.is_valid_number(parsed_number):
            details = {
                'time_zone': ", ".join(timezone.time_zones_for_number(parsed_number)),
                'carrier': carrier.name_for_number(parsed_number, 'en'),
                'region': geocoder.description_for_number(parsed_number, 'en')
            }
            return True, details
        else:
            return False, "Invalid phone number"
    except NumberParseException:
        return False, "NumberParseException was caught - the number might be in a wrong format or not valid."

def save_results(phone_number, details, filename="phone_number_details.csv"):
    """Save the phone number details to a CSV file."""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        fieldnames = ['phone_number', 'time_zone', 'carrier', 'region']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({'phone_number': phone_number, **details})

def gui():
    """Create a GUI for the phone number parser."""
    window = Tk()
    window.title("Phone Number Details Extractor")

    def on_submit():
        phone_number = entry.get()
        if not phone_number.strip():
            messagebox.showerror("Error", "Please enter a phone number.")
            return
        valid, details_or_message = parse_phone_number(phone_number)
        if valid:
            result_text = f"Time Zone(s): {details_or_message['time_zone']}\n"
            result_text += f"Region: {details_or_message['region']}\n"
            result_text += f"Carrier: {details_or_message['carrier']}\n"
            output.delete('1.0', END)
            output.insert('1.0', result_text)
        else:
            messagebox.showerror("Error", details_or_message)

    def save_to_file():
        phone_number = entry.get()
        valid, details_or_message = parse_phone_number(phone_number)
        if valid:
            filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
            if filename:
                save_results(phone_number, details_or_message, filename)
                messagebox.showinfo("Success", "Phone number details saved successfully.")
        else:
            messagebox.showerror("Error", "Invalid phone number or no phone number entered.")

    menu = Menu(window)
    window.config(menu=menu)
    file_menu = Menu(menu, tearoff=0)
    menu.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Save As...", command=save_to_file)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=window.quit)

    Label(window, text="Enter phone number:").pack()
    entry = Entry(window, width=50)
    entry.pack()
    Button(window, text="Submit", command=on_submit).pack()
    output = Text(window, height=10, width=50)
    output.pack()
    scrollbar = Scrollbar(window, command=output.yview, orient=VERTICAL)
    scrollbar.pack(side="right", fill="y")
    output.config(yscrollcommand=scrollbar.set)

    window.mainloop()

if __name__ == "__main__":
    gui()
