import json
import socket
from inspect import getmembers, ismethod
from threading import Thread

# Buffer size for receiving data from the socket
SIZE = 1024


class RPCServer:
    "A Remote Procedure Call (RPC) server that handles client requests and executes registered functions."

    def __init__(self, host: str = "0.0.0.0", port: int = 8033) -> None:
        "Initializes the RPC server with the provided host and port."
        self.address = (host, port)
        self._methods = {}  # Dictionary to store registered methods that can be called by the client

    def registerMethod(self, function) -> None:
        "Registers a single function/method for remote invocation by clients."
        try:
            self._methods.update({function.__name__: function})
        except Exception:
            raise Exception("Cannot register a non-function object")

    def registerInstance(self, instance=None) -> None:
        "Registers all methods of an instance of a class for remote invocation by clients."
        try:
            for functionName, function in getmembers(instance, predicate=ismethod):
                if not functionName.startswith("__"):
                    self._methods.update({functionName: function})
        except Exception:
            raise Exception("Cannot register a non-class object")

    def __handle__(self, client: socket.socket, address: tuple) -> None:
        "Handles incoming client requests by executing the requested function and sending back the result."

        print(f"Managing requests from {address}.")

        while True:
            try:
                # Deserialize JSON data into function name, arguments, and keyword arguments
                functionName, args, kwargs = json.loads(client.recv(SIZE).decode())
            except Exception:
                print(f"! Client {address} disconnected.")
                break

            print(f"> {address} : {functionName}({args})")

            try:
                response = self._methods[functionName](*args, **kwargs)
            except Exception as e:
                # Send back exception if function called by client is not registered
                client.sendall(json.dumps(str(e)).encode())
            else:
                client.sendall(json.dumps(response).encode())

            # Log the completion of handling the client's requests
        print(f"Completed requests from {address}.")
        client.close()

    def run(self) -> None:
        "Starts the RPC server and listens for incoming client connections."

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(self.address)
            sock.listen()

            print(f"+ Server {self.address} running")

            while True:
                try:
                    client, address = sock.accept()
                    Thread(target=self.__handle__, args=[client, address]).start()
                except KeyboardInterrupt:
                    print(f"- Server {self.address} interrupted")
                    break


class RPCClient:
    "A Remote Procedure Call (RPC) client that connects to an RPC server and invokes methods remotely."

    def __init__(self, host: str = "localhost", port: int = 8033) -> None:
        "Initializes the RPC client with the provided host and port."
        self.__sock: socket.socket | None = None
        self.__address = (host, port)

    def connect(self):
        "Connects the client to the RPC server."

        try:
            self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__sock.connect(self.__address)
        except EOFError as e:
            print(e)
            raise Exception("Client was not able to connect.")

    def disconnect(self):
        "Disconnects the client from the server."
        try:
            if self.__sock is not None:
                self.__sock.close()
        except Exception:
            # Ignore any exceptions that occur during disconnection
            pass

    def __getattr__(self, __name: str):
        "Dynamically handles method calls on the client by sending them to the server for execution."

        def execute(*args, **kwargs):
            if self.__sock is None:
                raise Exception("Client is not connected to the server")
            self.__sock.sendall(json.dumps((__name, args, kwargs)).encode())
            response = json.loads(self.__sock.recv(SIZE).decode())
            return response

        return execute
