import * as React from "react";

import { cn } from "@/lib/utils";

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "h-10 w-full min-w-0 rounded-md border border-[var(--line)] bg-[rgba(5,24,29,0.72)] px-3 py-2 text-sm text-[var(--ink)] outline-none transition-colors placeholder:text-[var(--muted)] disabled:cursor-not-allowed disabled:opacity-50 focus-visible:border-[var(--mint)] focus-visible:ring-2 focus-visible:ring-[rgba(128,226,187,0.18)] aria-invalid:border-[var(--coral)] aria-invalid:ring-[rgba(239,140,121,0.16)]",
        className,
      )}
      {...props}
    />
  );
}

export { Input };
