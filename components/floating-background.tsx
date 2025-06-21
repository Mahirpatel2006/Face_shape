"use client"

import { cn } from "@/lib/utils"

export default function FloatingBackground() {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden bg-zinc-100">
      <div
        className={cn(
          "absolute -top-1/2 left-1/2 h-[150%] w-1/2 -translate-x-1/2",
          "bg-gradient-to-br from-blue-100/80 via-blue-50/50 to-transparent",
          "animate-[aurora_12s_ease-in-out_infinite]"
        )}
      />
      <div
        className={cn(
          "absolute -top-1/3 left-1/2 h-[100%] w-2/3 -translate-x-1/4",
          "bg-gradient-to-br from-purple-100/80 via-purple-50/50 to-transparent",
          "animate-[aurora_15s_ease-in-out_infinite_3s]"
        )}
      />
      <div
        className={cn(
          "absolute -bottom-1/2 -left-1/4 h-[100%] w-2/3",
          "bg-gradient-to-br from-pink-100/80 via-pink-50/50 to-transparent",
          "animate-[aurora_18s_ease-in-out_infinite_6s]"
        )}
      />
      <div
        className={cn(
          "absolute -bottom-1/4 -right-1/4 h-[100%] w-1/2",
          "bg-gradient-to-br from-teal-100/80 via-teal-50/50 to-transparent",
          "animate-[aurora_20s_ease-in-out_infinite_9s]"
        )}
      />
      <div className="absolute inset-0 bg-white/20 backdrop-blur-sm" />
    </div>
  )
}
