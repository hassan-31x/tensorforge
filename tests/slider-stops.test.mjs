import assert from "node:assert/strict";
import test from "node:test";
import { sliderStops } from "../src/components/slider-stops.ts";

test("head and vocabulary sliders use meaningful discrete values", () => {
  const heads = sliderStops("Attention heads", 16);
  assert.deepEqual(heads.slice(0, 8), [1, 2, 4, 8, 12, 16, 24, 32]);
  assert.equal(heads.includes(17), false);

  const vocabulary = sliderStops("Vocabulary", 50000);
  assert.equal(vocabulary[vocabulary.indexOf(50000) + 1], 55000);
  assert.equal(vocabulary[vocabulary.indexOf(100000) + 1], 110000);
});

test("precise model preset values stay on the slider", () => {
  const vocabulary = sliderStops("Vocabulary size", 50257);
  assert.deepEqual(
    vocabulary.slice(
      vocabulary.indexOf(50257) - 1,
      vocabulary.indexOf(50257) + 2,
    ),
    [50000, 50257, 55000],
  );
  assert.equal(sliderStops("Learning rate", 0.00015).includes(0.00015), true);
});
