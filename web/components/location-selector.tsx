"use client";

import * as React from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

const locations = [
  {
    label: "Ethiopia",
    value: "ethiopia",
    coordinates: { latitude: 9, longitude: 39.5 },
  },
  {
    label: "Haiti",
    value: "haiti",
    coordinates: { latitude: 18.9712, longitude: -72.2852 },
  },
];

interface LocationSelectorProps {
  onLocationChange: (latitude: number, longitude: number) => void;
}

export function LocationSelector({ onLocationChange }: LocationSelectorProps) {
  const [open, setOpen] = React.useState(false);
  const [value, setValue] = React.useState("");

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between bg-white/5 border-white/10 text-white hover:bg-white/10 hover:text-white"
        >
          {value
            ? locations.find((location) => location.value === value)?.label
            : "Select location..."}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-full p-0">
        <Command>
          <CommandInput placeholder="Search location..." />
          <CommandEmpty>No location found.</CommandEmpty>
          <CommandGroup>
            {locations.map((location) => (
              <CommandItem
                key={location.value}
                value={location.value}
                onSelect={(currentValue) => {
                  setValue(currentValue === value ? "" : currentValue);
                  const selectedLocation = locations.find(
                    (loc) => loc.value === currentValue
                  );
                  if (selectedLocation) {
                    onLocationChange(
                      selectedLocation.coordinates.latitude,
                      selectedLocation.coordinates.longitude
                    );
                  }
                  setOpen(false);
                }}
              >
                <Check
                  className={cn(
                    "mr-2 h-4 w-4",
                    value === location.value ? "opacity-100" : "opacity-0"
                  )}
                />
                {location.label}
              </CommandItem>
            ))}
          </CommandGroup>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
