import phonenumbers
from phonenumbers import timezone
from phonenumbers import geocoder, carrier
from phonenumbers import is_carrier_specific_for_region
import time
print("You need to enter the Target area code")

time.sleep(2)

phoneNumber = phonenumbers.parse(input("Type here the Target phone number: "))

timeZone = timezone.time_zones_for_number(phoneNumber)

choose_language1 = print("Format has to be 'en'")

choose_language = input("Type here the language you want: ")

Carrier = carrier.name_for_number(phoneNumber, choose_language)

Carrier_is_specific_for_carrier_region = is_carrier_specific_for_region(phoneNumber, choose_language)

Region = geocoder.description_for_number(phoneNumber, choose_language)

valid = phonenumbers.is_valid_number(phoneNumber)

possible = phonenumbers.is_possible_number(phoneNumber)

print(timeZone)
print(Carrier)
print(phoneNumber)
print(Region)
print(valid)
print(possible)
pass
