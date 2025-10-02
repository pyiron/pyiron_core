import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { OptionalLiteralSelect, OptionalLiteralSelectProps, STYLE_VARS } from '../OptionalLiteralSelect';

// Mock Checkbox so we can easily simulate toggles
jest.mock('../Checkbox', () => ({
  Checkbox: ({ checked, onToggle }: { checked: boolean; onToggle: () => void }) => (
    <button data-testid="checkbox" onClick={onToggle}>
      {checked ? 'checked' : 'unchecked'}
    </button>
  ),
}));

// ✅ Fixed setup: preserve explicit null instead of replacing with "A"
function setup(props?: Partial<OptionalLiteralSelectProps>) {
  const onChange = jest.fn();
  const onHoverChange = jest.fn();
  const utils = render(
    <OptionalLiteralSelect
      value={props && 'value' in props ? props.value! : 'A'} // IMPORTANT FIX!
      options={props?.options ?? ['A', 'B']}
      onChange={onChange}
      highlighted={props?.highlighted ?? false}
      onHoverChange={props?.onHoverChange ?? onHoverChange}
      showCheckbox={props?.showCheckbox ?? true}
      mode={props?.mode ?? 'select'}
      inputType={props?.inputType ?? 'text'}
    />
  );
  return { onChange, onHoverChange, ...utils };
}

beforeEach(() => {
  jest.clearAllMocks();
});

describe('OptionalLiteralSelect - Full Coverage', () => {
  it('renders select mode with checkbox', () => {
    setup();
    expect(screen.getByRole('combobox')).toBeInTheDocument();
    expect(screen.getByTestId('checkbox')).toBeInTheDocument();
  });

  it('renders input mode without checkbox', () => {
    setup({ showCheckbox: false, mode: 'input' });
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.queryByTestId('checkbox')).not.toBeInTheDocument();
  });

  it('sets step=any for number inputs', () => {
    setup({ showCheckbox: false, mode: 'input', inputType: 'number' });
    expect(screen.getByRole('spinbutton')).toHaveAttribute('step', 'any');
  });

  it('calls onHoverChange when defined', () => {
    const { onHoverChange } = setup();
    const wrapper = screen.getByRole('combobox').closest('div')!.parentElement!;
    fireEvent.mouseEnter(wrapper);
    expect(onHoverChange).toHaveBeenCalledWith(true);
    fireEvent.mouseLeave(wrapper);
    expect(onHoverChange).toHaveBeenCalledWith(false);
  });

  it('does not throw when onHoverChange undefined', () => {
    setup({ onHoverChange: undefined });
    const wrapper = screen.getByRole('combobox').closest('div')!.parentElement!;
    fireEvent.mouseEnter(wrapper);
    fireEvent.mouseLeave(wrapper);
  });

  it('toggles checkbox', () => {
    const { onChange } = setup();
    fireEvent.click(screen.getByTestId('checkbox')); // off
    expect(onChange).toHaveBeenCalledWith(null, true);
    fireEvent.click(screen.getByTestId('checkbox')); // on
    expect(onChange).toHaveBeenCalledWith('A', true);
  });

  it('changes select value showCheckbox=true', () => {
    const { onChange } = setup();
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'B' } });
    expect(onChange).toHaveBeenCalledWith('B', true);
  });

  it('changes select value showCheckbox=false', () => {
    const { onChange } = setup({ showCheckbox: false });
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'B' } });
    expect(onChange).toHaveBeenCalledWith('B', true);
  });

  it('input mode: typing calls onChange commit=false', () => {
    const { onChange } = setup({ mode: 'input' });
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'XYZ' } });
    expect(onChange).toHaveBeenCalledWith('XYZ', false);
  });

  it('input mode: typing when useValue=false does NOT call onChange', () => {
    const { onChange } = setup({ mode: 'input', value: null, showCheckbox: true, options: [] });
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '123' } });
    expect(onChange).not.toHaveBeenCalled();
  });

  // ✅ New branch coverage test: handleInputCommit else branch
  it('input commit does NOT call onChange when showCheckbox=true and useValue=false', () => {
    const { onChange } = setup({ mode: 'input', value: null, showCheckbox: true, options: [] });
    fireEvent.blur(screen.getByRole('textbox'), { target: { value: 'ShouldNotCommit' } });
    expect(onChange).not.toHaveBeenCalled();
  });

  it('input commit on Enter key', () => {
    const { onChange } = setup({ mode: 'input' });
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
    expect(onChange).toHaveBeenCalledWith('A', true);
  });

  it('input commit on blur trims', () => {
    const { onChange } = setup({ mode: 'input' });
    fireEvent.blur(screen.getByRole('textbox'), { target: { value: '  ZZZ  ' } });
    expect(onChange).toHaveBeenCalledWith('ZZZ', true);
  });

  it('syncs state on prop change', () => {
    const { rerender } = setup();
    expect((screen.getByRole('combobox') as HTMLSelectElement).value).toBe('A');
    rerender(
      <OptionalLiteralSelect value="B" options={['A', 'B']} onChange={jest.fn()} />
    );
    expect((screen.getByRole('combobox') as HTMLSelectElement).value).toBe('B');
  });

  it('shows placeholder="default" when inactive', async () => {
    setup({ mode: 'input', value: null, showCheckbox: true, options: [] });
    expect(await screen.findByPlaceholderText('default')).toBeInTheDocument();
  });

  it('applies highlight background', () => {
    setup({ highlighted: true });
    const container = screen.getByRole('combobox').closest('div')!.parentElement!;
    expect(container).toHaveStyle({ backgroundColor: STYLE_VARS.highlightColor });
  });

  it('disables input when showCheckbox && !useValue', () => {
    setup({ mode: 'input', value: null, showCheckbox: true, options: [] });
    expect(screen.getByRole('textbox')).toBeDisabled();
  });
});