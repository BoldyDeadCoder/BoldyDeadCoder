import phonenumbers
from phonenumbers import timezone, geocoder, carrier, NumberParseException
import csv
import sys
import os

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

def main_loop():
    """Main loop for processing phone numbers with an option to exit."""
    while True:
        phone_number = input("Type here the Target phone number or 'exit' to quit: ").strip()
        if phone_number.lower() == 'exit':
            print("Exiting program.")
            break
        elif phone_number:
            valid, details_or_message = parse_phone_number(phone_number)
            if valid:
                print("Time Zone(s):", details_or_message['time_zone'])
                print("Region:", details_or_message['region'])
                print("Carrier:", details_or_message['carrier'])
                save_results(phone_number, details_or_message)
                print(f"Details for {phone_number} have been saved.")
            else:
                print(details_or_message)
        else:
            print("No phone number was entered. Please try again.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Handle phone number passed as command-line argument
        phone_number = sys.argv[1]
        valid, details_or_message = parse_phone_number(phone_number)
        if valid:
            print("Time Zone(s):", details_or_message['time_zone'])
            print("Region:", details_or_message['region'])
            print("Carrier:", details_or_message['carrier'])
            save_results(phone_number, details_or_message)
            print(f"Details for {phone_number} have been saved.")
        else:
            print(details_or_message)
    else:
        # Interactive mode
        main_loop()
