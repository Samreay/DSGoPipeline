from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
from pyftpdlib.authorizers import DummyAuthorizer
import os

def main():
    authorizer = DummyAuthorizer()
    authorizer.add_user('user', '12345', '.', perm='elradfmwMT')
    authorizer.add_anonymous(os.getcwd())
    
    handler = FTPHandler
    handler.authorizer = authorizer
    server = FTPServer(('127.0.0.1', 2121), handler)
    server.serve_forever()

if __name__ == "__main__":
    main()