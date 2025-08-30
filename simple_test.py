import backend

print("Testing path validation...")
try:
    backend._validate_path("../../../etc/passwd")
    print("ERROR: Path traversal not blocked")
except ValueError as e:
    print("SUCCESS: Path traversal blocked -", str(e))

print("\nTesting user operations...")
users = backend.get_registered_users()
print("Current users:", users)

print("\nTesting authentication error handling...")
result = backend.authenticate("nonexistent_user", "fake_file.csv")
print("Non-existent user result:", result)