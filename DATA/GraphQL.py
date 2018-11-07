from ariadne import GraphQLMiddleware

# Define types using Schema Definition Language (https://graphql.org/learn/schema/)
type_defs = """
    type Query {
        people: [Person!]!
    }

    type Person {
        firstName: String
        lastName: String
        age: Int
        fullName: String
    }
"""


# Resolvers are simple python functions
def resolve_people(*_):
    return [
        {"firstName": "John", "lastName": "Doe", "age": 21},
        {"firstName": "Bob", "lastName": "Boberson", "age": 24},
    ]


def resolve_person_fullname(person, *_):
    return "%s %s" % (person["firstName"], person["lastName"])


# Map resolver functions to type fields using dict
resolvers = {
    "Query": {"people": resolve_people},
    "Person": {"fullName": resolve_person_fullname},
}


# Create and run dev server that provides api browser
graphql_server = GraphQLMiddleware.make_simple_server(type_defs, resolvers)
graphql_server.serve_forever()  # Visit http://127.0.0.1:8888 to see API browser!
