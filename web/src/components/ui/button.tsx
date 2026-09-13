import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors outline-none disabled:pointer-events-none disabled:opacity-50 focus-visible:ring-2 focus-visible:ring-[var(--gold)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--deep)]",
  {
    variants: {
      variant: {
        default:
          "border border-[var(--mint)] bg-[var(--mint)] text-[var(--deep)] hover:bg-[#a1eccf]",
        outline:
          "border border-[var(--line)] bg-transparent text-[var(--ink)] hover:border-[rgba(128,226,187,0.65)] hover:bg-[rgba(128,226,187,0.09)]",
        ghost:
          "border border-transparent bg-transparent text-[var(--muted)] hover:text-[var(--ink)]",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

function Button({
  className,
  variant,
  size,
  ...props
}: React.ComponentProps<"button"> & VariantProps<typeof buttonVariants>) {
  return (
    <button
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button };
