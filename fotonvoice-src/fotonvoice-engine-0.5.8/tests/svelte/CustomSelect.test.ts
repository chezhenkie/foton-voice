import { describe, test, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/svelte";
import CustomSelect from "../../src/lib/Settings/CustomSelect.svelte";

const options = [
  { value: "a", label: "Option A" },
  { value: "b", label: "Option B" },
  { value: "c", label: "Option C" },
];

/**
 * jsdom lays nothing out, so the trigger's box is whatever we say it is. This
 * is what decides where the menu goes.
 */
function stubTriggerRect(top: number, height = 36) {
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
    top,
    bottom: top + height,
    left: 20,
    right: 320,
    width: 300,
    height,
    x: 20,
    y: top,
    toJSON: () => ({}),
  } as DOMRect);
}

async function openMenu(container: HTMLElement) {
  const trigger = container.querySelector(".custom-select-trigger") as HTMLElement;
  await fireEvent.click(trigger);
  return container.querySelector(".custom-dropdown-menu") as HTMLElement;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("CustomSelect dropdown placement", () => {
  test("shows every option when opened", async () => {
    stubTriggerRect(100);
    const { container } = render(CustomSelect, { value: "a", options });
    const menu = await openMenu(container);

    for (const opt of options) {
      expect(within(menu).getByText(opt.label)).not.toBeNull();
    }
  });

  /**
   * The settings panels and the target editor scroll their content, and an
   * absolutely positioned menu is clipped by that overflow - the bug that hid
   * half the delivery list. Fixed positioning is what escapes the clip.
   */
  test("positions the menu with fixed coordinates so a scrolling panel cannot clip it", async () => {
    stubTriggerRect(100);
    const { container } = render(CustomSelect, { value: "a", options });
    const menu = await openMenu(container);

    expect(menu.style.position).toBe("fixed");
    expect(menu.style.left).toBe("20px");
    expect(menu.style.width).toBe("300px");
    expect(menu.style.top).toBe("140px");
    expect(menu.style.bottom).toBe("");
  });

  test("caps the menu height to the space it has on screen", async () => {
    stubTriggerRect(100);
    const { container } = render(CustomSelect, { value: "a", options });
    const menu = await openMenu(container);

    const maxHeight = parseInt(menu.style.maxHeight, 10);
    expect(maxHeight).toBeGreaterThan(0);
    expect(maxHeight).toBeLessThanOrEqual(window.innerHeight);
  });

  test("flips above the trigger when there is no room below", async () => {
    stubTriggerRect(window.innerHeight - 60);
    const { container } = render(CustomSelect, { value: "a", options });
    const menu = await openMenu(container);

    expect(menu.style.position).toBe("fixed");
    expect(menu.style.bottom).not.toBe("");
    expect(menu.style.top).toBe("");
  });

  test("selecting an option closes the menu and reports the value", async () => {
    stubTriggerRect(100);
    const onchange = vi.fn();
    const { container } = render(CustomSelect, { value: "a", options, onchange });
    await openMenu(container);

    await fireEvent.click(screen.getByText("Option B"));

    expect(onchange).toHaveBeenCalledWith("b");
    expect(container.querySelector(".custom-dropdown-menu")).toBeNull();
  });

  test("offsets fixed coordinates when an ancestor forms a containing block", async () => {
    stubTriggerRect(100);
    const { container } = render(CustomSelect, { value: "a", options });

    vi.spyOn(window, "getComputedStyle").mockImplementation((el: Element) => {
      if (el === container) {
        return {
          transform: "translateY(0px)",
          perspective: "none",
          filter: "none",
          contain: "none",
          willChange: "auto",
        } as CSSStyleDeclaration;
      }
      return {
        transform: "none",
        perspective: "none",
        filter: "none",
        contain: "none",
        willChange: "auto",
      } as CSSStyleDeclaration;
    });

    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
      if (this === container) {
        return {
          top: 40,
          bottom: 600,
          left: 10,
          right: 500,
          width: 490,
          height: 560,
          x: 10,
          y: 40,
          toJSON: () => ({}),
        } as DOMRect;
      }
      return {
        top: 100,
        bottom: 136,
        left: 20,
        right: 320,
        width: 300,
        height: 36,
        x: 20,
        y: 100,
        toJSON: () => ({}),
      } as DOMRect;
    });

    const menu = await openMenu(container);

    expect(menu.style.position).toBe("fixed");
    expect(menu.style.left).toBe("10px");
    expect(menu.style.top).toBe("100px");
  });

  test("defaults to the first option when defaultToFirst is true and value is unset", async () => {
    stubTriggerRect(100);
    const onchange = vi.fn();
    const { container } = render(CustomSelect, { value: "", options, defaultToFirst: true, onchange });

    const trigger = container.querySelector(".custom-select-trigger") as HTMLElement;
    expect(trigger.textContent).toContain("Option A");
    expect(onchange).toHaveBeenCalledWith("a");
  });

  test("defaults to the first option when defaultToFirst is true and value is not in options", async () => {
    stubTriggerRect(100);
    const onchange = vi.fn();
    const { container } = render(CustomSelect, { value: "non_existent_voice", options, defaultToFirst: true, onchange });

    const trigger = container.querySelector(".custom-select-trigger") as HTMLElement;
    expect(trigger.textContent).toContain("Option A");
    expect(onchange).toHaveBeenCalledWith("a");
  });

  test("preserves valid value when defaultToFirst is true", async () => {
    stubTriggerRect(100);
    const onchange = vi.fn();
    const { container } = render(CustomSelect, { value: "b", options, defaultToFirst: true, onchange });

    const trigger = container.querySelector(".custom-select-trigger") as HTMLElement;
    expect(trigger.textContent).toContain("Option B");
    expect(onchange).not.toHaveBeenCalled();
  });
});
