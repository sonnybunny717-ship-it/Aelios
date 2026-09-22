const OPEN_TAG = "<thinking>";
const CLOSE_TAG = "</thinking>";
const TAGS = [OPEN_TAG, CLOSE_TAG];

type Source = "text" | "thinking";
type RoutedDelta = { textDelta: string; thinkingDelta: string };
type Token = { char: string; source: Source };

function appendRouted(output: RoutedDelta, token: Token, insideThinking: boolean, textAfterThinking: boolean): void {
  const target = insideThinking || (!textAfterThinking && token.source === "thinking")
    ? "thinkingDelta"
    : "textDelta";
  output[target] += token.char;
}

/**
 * Keep persistence aligned with the garden stream parser. Some compatible
 * providers leave `</thinking>` and the visible answer in reasoning_content.
 */
export function createThinkingStreamParser() {
  let insideThinking = false;
  let textAfterThinking = false;
  let pending: Token[] = [];
  let finished = false;

  function push(value: string, source: Source): RoutedDelta {
    if (finished || !value) return { textDelta: "", thinkingDelta: "" };

    const output: RoutedDelta = { textDelta: "", thinkingDelta: "" };
    const tokens = [...pending, ...Array.from(value, char => ({ char, source }))];
    pending = [];
    const lower = tokens.map(token => token.char).join("").toLowerCase();

    let index = 0;
    while (index < tokens.length) {
      if (lower.startsWith(OPEN_TAG, index)) {
        insideThinking = true;
        textAfterThinking = false;
        index += OPEN_TAG.length;
        continue;
      }
      if (lower.startsWith(CLOSE_TAG, index)) {
        insideThinking = false;
        textAfterThinking = true;
        index += CLOSE_TAG.length;
        continue;
      }

      const remainder = lower.slice(index);
      if (TAGS.some(tag => tag.startsWith(remainder))) {
        pending = tokens.slice(index);
        break;
      }

      appendRouted(output, tokens[index], insideThinking, textAfterThinking);
      index += 1;
    }

    return output;
  }

  function finish(): RoutedDelta {
    if (finished) return { textDelta: "", thinkingDelta: "" };
    finished = true;
    const output: RoutedDelta = { textDelta: "", thinkingDelta: "" };
    const tail = pending;
    pending = [];
    const raw = tail.map(token => token.char).join("");
    if (/^<\/?think/i.test(raw)) return output;
    for (const token of tail) appendRouted(output, token, insideThinking, textAfterThinking);
    return output;
  }

  return {
    pushText: (value: string) => push(value, "text"),
    pushThinking: (value: string) => push(value, "thinking"),
    finish,
  };
}
