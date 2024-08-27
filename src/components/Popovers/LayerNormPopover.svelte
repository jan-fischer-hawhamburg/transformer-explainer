<script lang="ts">
	import classNames from 'classnames';
	import { Popover } from 'flowbite-svelte';
	import type { PopoverProps } from 'flowbite-svelte/Popover.svelte';
	import { fade } from 'svelte/transition';
	import CommonPopover from './CommonPopover.svelte';

	export let offset: PopoverProps['offset'] = undefined;
	export let className: PopoverProps['class'] = undefined;
	export let triggeredBy: PopoverProps['triggeredBy'] = undefined;
	export let trigger: PopoverProps['trigger'] = undefined;
	export let placement: PopoverProps['placement'] = undefined;
</script>

<CommonPopover
	className="ln-popover"
	title="Layer Normalization"
	{offset}
	{triggeredBy}
	{trigger}
	{placement}
	goTo="article-ln"
>
	<div class="ln-content w-[10rem]">
		Standardisiert die Layer-Eingaben (layer inputs), um Mittelwert und Varianz konsistent zu halten.
	</div></CommonPopover
>
