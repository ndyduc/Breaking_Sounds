import argparse
import hashlib
import os
import re
import socket
import sys
from urllib.parse import unquote

# If using Web flow, the redirect URL must match exactly what’s configured in GCP for
# the OAuth client.  If using Desktop flow, the redirect must be a localhost URL and
# is not explicitly set in GCP.
from google_auth_oauthlib.flow import Flow

_SCOPE = "https://www.googleapis.com/auth/adwords"
_SERVER = "127.0.0.1"
_PORT = 8080
_REDIRECT_URI = f"http://127.0.0.1:3202/"


def main(client_secrets_path, scopes):
    flow = Flow.from_client_secrets_file(client_secrets_path, scopes=scopes)
    flow.redirect_uri = _REDIRECT_URI

    passthrough_val = hashlib.sha256(os.urandom(1024)).hexdigest()

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        state=passthrough_val,
        prompt="consent",
        include_granted_scopes="true",
    )
    print("Paste this URL into your browser: ")
    print(authorization_url)
    print(f"\nWaiting for authorization and callback to: {_REDIRECT_URI}")

    # Retrieves an authorization code by opening a socket to receive the
    # redirect request and parsing the query parameters set in the URL.
    code = unquote(get_authorization_code(passthrough_val))

    # Pass the code back into the OAuth module to get a refresh token.
    flow.fetch_token(code=code)
    refresh_token = flow.credentials.refresh_token

    print(f"\nYour refresh token is: {refresh_token}\n")
    print(
        "Add your refresh token to your client library configuration as "
        "described here: "
        "https://developers.google.com/google-ads/api/docs/client-libs/python/configuration"
    )


def get_authorization_code(passthrough_val):
    # Open a socket at _SERVER:_PORT and listen for a request
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((_SERVER, _PORT))
    sock.listen(1)
    connection, address = sock.accept()
    data = connection.recv(1024)
    # Parse the raw request to retrieve the URL query parameters.
    params = parse_raw_query_params(data)

    try:
        if not params.get("code"):
            # If no code is present in the query params then there will be an
            # error message with more details.
            error = params.get("error")
            message = f"Failed to retrieve authorization code. Error: {error}"
            raise ValueError(message)
        elif params.get("state") != passthrough_val:
            message = "State token does not match the expected state."
            raise ValueError(message)
        else:
            message = "Authorization code was successfully retrieved."
    except ValueError as error:
        print(error)
        sys.exit(1)
    finally:
        response = (
            "HTTP/1.1 200 OK\n"
            "Content-Type: text/html\n\n"
            f"<b>{message}</b>"
            "<p>Please check the console output.</p>\n"
        )

        connection.sendall(response.encode())
        connection.close()

    return params.get("code")


def parse_raw_query_params(data):
    """Parses a raw HTTP request to extract its query params as a dict.

    Note that this logic is likely irrelevant if you're building OAuth logic
    into a complete web application, where response parsing is handled by a
    framework.

    Args:
        data: raw request data as bytes.

    Returns:
        a dict of query parameter key value pairs.
    """
    # Decode the request into a utf-8 encoded string
    decoded = data.decode("utf-8")
    # Use a regular expression to extract the URL query parameters string
    match = re.search(r"GET\s\/\?(.*) ", decoded)
    params = match.group(1)
    # Split the parameters to isolate the key/value pairs
    pairs = [pair.split("=") for pair in params.split("&")]
    # Convert pairs to a dict to make it easy to access the values
    return {key: val for key, val in pairs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generates OAuth2 refresh token using the Web application flow. "
            "To retrieve the necessary client_secrets JSON file, first "
            "generate OAuth 2.0 credentials of type Web application in the "
            "Google Cloud Console (https://console.cloud.google.com). "
            "Make sure 'http://_SERVER:_PORT' is included the list of "
            "'Authorized redirect URIs' for this client ID."
        ),
    )
    # The following argument(s) should be provided to run the example.
    parser.add_argument(
        "-c",
        "--client_secrets_path",
        required=True,
        type=str,
        help=(
            "Path to the client secrets JSON file from the Google Developers "
            "Console that contains your client ID, client secret, and "
            "redirect URIs."
        ),
    )
    parser.add_argument(
        "--additional_scopes",
        default=None,
        type=str,
        nargs="+",
        help="Additional scopes to apply when generating the refresh token.",
    )
    args = parser.parse_args()

    configured_scopes = [_SCOPE]

    if args.additional_scopes:
        configured_scopes.extend(args.additional_scopes)

    main(args.client_secrets_path, configured_scopes)
