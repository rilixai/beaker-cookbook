---
name: notion
description: Procedures for Notion pages, including creating complete text pages with a parent, title and body.
---

For a new text page, gather the required content before creating it. Call `api_fetch` with endpoint `notion.pages.create`. Pass `body` as a JSON object, not a quoted JSON string. Include the destination parent, title property, and page body in the same call. A title alone does not populate the page body.

For ordinary text, use the supported `markdown` field to avoid constructing nested block arrays. Example argument shape (replace the placeholders with the actual values):

```json
{
  "endpoint": "notion.pages.create",
  "body": {
    "parent": {"page_id": "<destination page ID>"},
    "properties": {
      "title": {"title": [{"text": {"content": "<page title>"}}]}
    },
    "markdown": "<complete page body>"
  }
}
```

Use `database_id` instead of `page_id` only when the destination is a database. Preserve supplied names and source values in the title and body. A valid `children` array remains suitable when structured blocks are needed; use one content representation per page.

If validation fails, correct the argument structure while retaining the required parent, title and body. Do not retry with an empty body or discard content to obtain success. `notion.pages.update` changes page properties and archive state; it does not append or replace body content. Its success cannot repair a page created without the required body. After a successful creation, use the returned page ID and URL for subsequent actions instead of creating duplicate pages.
