from piculet import reducers
{
"path": ".//title/text()",
    "reduce": reducers.first
}
items = [
        {
            "key": "full_title",
            "value": {
                "path": ".//h1//text()",
                "reduce": reducers.join
            }
        }
    ]
