"""
Read a relaxed JSON file.

Relaxed JSON allows comments (introduced by // and going to the end
of the line), optional quotes on key names in dictionaries and optional
trailing commas at the ends of lists and dictionaries.  It also strips
the leading characters up to the first '{'.  Multiline strings can be
formatted using "\n\" at the end of each line.

If the file contains e.g., "var _ = {...}", then it can be edited with
a JavaScript aware editor such as VJET for eclipse, and it will be easier
to locate errors and adjust formatting.
"""
import re
import json

_LEADING_TEXT = re.compile(r'^.*?[{]', re.DOTALL)
_LINE_CONTINUATION = re.compile(r'\\\s*\n')
_TRAILING_COMMENT = re.compile(r'(?P<comment>\s*//.*?)\n')
_MULTILINE_COMMENT = re.compile(r'(?P<comment>/\*.*?\*/)', re.DOTALL)
_UNQUOTED_FIELDNAME = re.compile(r'(?P<prefix>[,{]\s*)(?P<key>[^\s,{}:"]+)(?P<tail>\s*:)')
_TRAILING_COMMA = re.compile(r',(?P<tail>\s*[]}])')

def load(path, **kw):
    return loads(open(path).read(), **kw)

def loads(text, **kw):
    """
    Parse and return a relaxed JSON string.
    """
    # TODO: need a little state machine that performs the translation so that
    # TODO: line and column numbers are preserved, and so that we can have
    # TODO: http:// in a string (instead of it being treated like a comment).
    #print "== raw text\n", text
    text = _LINE_CONTINUATION.sub('', text)
    #print "== joined lines\n", text
    text = _TRAILING_COMMENT.sub(r'\n', text)
    text = _MULTILINE_COMMENT.sub(r'', text)
    #print "== stripped comments\n", text
    text = _LEADING_TEXT.sub('{', text)
    #print "== trimmed text\n", text
    text = _UNQUOTED_FIELDNAME.sub(r'\g<prefix>"\g<key>"\g<tail>', text)
    #print "== quoted field names\n", text
    text = _TRAILING_COMMA.sub(r'\g<tail>', text)
    #print "== processed text\n", text
    try:
        #from ..writer import util
        #obj = json.loads(text, object_hook=util.json_decode_dict_as_str, **kw)
        obj = json.loads(text, **kw)
    except ValueError as e:
        msg = [str(e)]
        M = re.findall('line ([0-9]*) column ([0-9]*)', msg[0])
        if M:
            line, col = int(M[0][0]), int(M[0][1])
            lines = text.split("\n")
            if line >= 2:
                msg.append(lines[line-2])
            if line >= 1:
                msg.append(lines[line-1])
            msg.append(" "*(col-1) + "^")
            if line < len(lines):
                msg.append(lines[line])
        msg = "\n".join(msg)
        raise ValueError(msg)
    return obj


def test():
    """
    Verify that the translation from pseudo-JSON to JSON works.
    """
    good = """\
// This is a source definition with no errors
var entry = {
field : { // A comment about the field
  "field" : "te\\
x\\
t",
  other$field : 56,
  },
/*
multiline comment
*/
secondfield : {
  content: ["string", "string"], /* a second comment */
  content: [{name:"good", value:3, URL:"http:\\/\\/my.url.com"},]
  },
}
"""
    broken = """\
// This is a source definition with a missing comma
{
field : { // A comment about the field
  field : "te\\
x\\
t"
  other$field : 56,
  },
/*
multiline comment
*/
secondfield : {
  content: ["string", "string"], /* a second comment */
  content: [{name:"good", value:3},]
  },
}
"""
    result = loads(good)
    assert result['field']['field'] == "text"
    assert result['field']['other$field'] == 56
    assert result['secondfield']['content'][0]['name'] == 'good'
    try:
        loads(broken)
    except ValueError as _:
        pass
    else:
        raise Exception("No exception raised in broken")

if __name__ == "__main__":
    test()
