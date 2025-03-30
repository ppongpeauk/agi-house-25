"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Inter } from "next/font/google";
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query";
import { DiseaseRiskPanel } from "../components/disease-risk-panel";
import { LocationSelector } from "../components/location-selector";
import { Calendar } from "@/components/ui/calendar";
import { Button } from "@/components/ui/button";
import { CalendarIcon } from "lucide-react";
import { format } from "date-fns";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

const inter = Inter({ subsets: ["latin"] });

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Prediction {
  disease_name: string;
  probability: number;
}

interface WeatherData {
  temperature: number;
  humidity: number;
  wind_speed: number;
}

interface InferenceResponse {
  predictions: Prediction[];
  selected_date_weather: WeatherData;
  location: {
    latitude: number;
    longitude: number;
  };
}

const DynamicMap = dynamic(() => import("../components/Map"), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 flex items-center justify-center bg-black/20 backdrop-blur-sm">
      <div className="w-12 h-12 bg-cyan-500 rounded-full animate-pulse" />
    </div>
  ),
});

// Create a client
const queryClient = new QueryClient();

function HomePage() {
  const [date, setDate] = useState<Date>(new Date());
  const [coordinates, setCoordinates] = useState({
    latitude: 9,
    longitude: 39.5,
  });

  const { data, isLoading, error } = useQuery<InferenceResponse>({
    queryKey: ["inference", date, coordinates.latitude, coordinates.longitude],
    queryFn: async () => {
      try {
        const response = await fetch(`${API_URL}/inference`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({
            latitude: coordinates.latitude,
            longitude: coordinates.longitude,
            timestamp: Math.floor(date.getTime() / 1000),
          }),
        });

        if (!response.ok) {
          throw new Error(`API Error: ${response.status}`);
        }

        return response.json();
      } catch (err) {
        console.error("API Error:", err);
        throw new Error(
          "Failed to fetch disease predictions. Please try again later."
        );
      }
    },
    retry: 2,
  });

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      {/* Navigation */}
      <nav className="h-14 border-b border-white/10 bg-black/50 backdrop-blur-lg flex items-center px-6 sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold">
            <span className="text-cyan-500">DISEASE</span>{" "}
            <span className="text-white">INTELLIGENCE</span>
          </h1>
          <div className="px-2 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-500">
            LIVE
          </div>
        </div>
        <div className="ml-auto text-sm text-white/70">
          {new Date().toLocaleTimeString()}
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 flex">
        {/* Map Section */}
        <div className="relative flex-1 bg-zinc-900">
          <DynamicMap
            latitude={coordinates.latitude}
            longitude={coordinates.longitude}
          />

          {/* Overlay Cards */}
          <div className="absolute top-4 left-4 space-y-2 z-10">
            {/* Location Selector */}
            <div className="w-56 bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-4">
              <div className="flex items-center gap-2 text-xs text-white/60 mb-2">
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                LOCATION
              </div>
              <LocationSelector
                onLocationChange={(latitude, longitude) =>
                  setCoordinates({ latitude, longitude })
                }
              />
            </div>

            {/* Date Picker */}
            <div className="w-56 bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-4">
              <div className="flex items-center gap-2 text-xs text-white/60 mb-2">
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                DATE
              </div>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-full justify-start text-left font-normal bg-white/5 border-white/10 text-white hover:bg-white/10 hover:text-white",
                      !date && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {date ? format(date, "PPP") : <span>Pick a date</span>}
                  </Button>
                </PopoverTrigger>
                <PopoverContent
                  className="w-[280px] p-0"
                  style={{
                    backgroundColor: "rgba(0, 0, 0, 0.9)",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                  }}
                >
                  <Calendar
                    mode="single"
                    selected={date}
                    onSelect={(newDate) => newDate && setDate(newDate)}
                    initialFocus
                    className="bg-transparent text-white"
                    classNames={{
                      months:
                        "flex flex-col sm:flex-row space-y-4 sm:space-x-4 sm:space-y-0",
                      month: "space-y-4",
                      caption:
                        "flex justify-center pt-1 relative items-center text-white",
                      caption_label: "text-sm font-medium text-white",
                      nav: "space-x-1 flex items-center",
                      nav_button: cn(
                        "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100 text-white border border-white/10"
                      ),
                      table: "w-full border-collapse space-y-1",
                      head_row: "flex",
                      head_cell:
                        "text-white/60 rounded-md w-9 font-normal text-[0.8rem]",
                      row: "flex w-full mt-2",
                      cell: "text-center text-sm p-0 relative [&:has([aria-selected].day-range-end)]:rounded-r-md [&:has([aria-selected].day-outside)]:bg-accent/50 [&:has([aria-selected])]:bg-accent first:[&:has([aria-selected])]:rounded-l-md last:[&:has([aria-selected])]:rounded-r-md focus-within:relative focus-within:z-20",
                      day: cn(
                        "h-9 w-9 p-0 font-normal text-white aria-selected:opacity-100 hover:bg-white/10"
                      ),
                      day_range_end: "day-range-end",
                      day_selected:
                        "bg-cyan-500 text-white hover:bg-cyan-500 hover:text-white focus:bg-cyan-500 focus:text-white",
                      day_today: "bg-white/5 text-white",
                      day_outside:
                        "text-white/40 opacity-50 aria-selected:bg-accent/50 aria-selected:text-white aria-selected:opacity-30",
                      day_disabled: "text-white/20",
                      day_range_middle:
                        "aria-selected:bg-accent aria-selected:text-accent-foreground",
                      day_hidden: "invisible",
                    }}
                  />
                </PopoverContent>
              </Popover>
            </div>

            {/* Weather Cards */}
            <div className="w-56 bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-4 group hover:bg-black/90 transition-colors">
              <div className="flex items-center gap-2 text-xs text-white/60 mb-1">
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                TEMPERATURE
              </div>
              <div className="text-3xl font-bold text-cyan-500 flex items-baseline gap-1">
                {data?.selected_date_weather?.temperature.toFixed(1)}
                <span className="text-sm text-white/40">°C</span>
              </div>
              <div className="h-1 bg-white/5 rounded-full mt-3 overflow-hidden">
                <div className="h-full w-3/4 bg-gradient-to-r from-cyan-500 to-pink-500 rounded-full" />
              </div>
            </div>

            <div className="w-56 bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-4 group hover:bg-black/90 transition-colors">
              <div className="flex items-center gap-2 text-xs text-white/60 mb-1">
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                HUMIDITY
              </div>
              <div className="text-3xl font-bold text-cyan-500 flex items-baseline gap-1">
                {data?.selected_date_weather?.humidity.toFixed(0)}
                <span className="text-sm text-white/40">%</span>
              </div>
              <div className="h-1 bg-white/5 rounded-full mt-3 overflow-hidden">
                <div className="h-full w-2/3 bg-gradient-to-r from-cyan-500 to-pink-500 rounded-full" />
              </div>
            </div>

            <div className="w-56 bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-4 group hover:bg-black/90 transition-colors">
              <div className="flex items-center gap-2 text-xs text-white/60 mb-1">
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                WIND SPEED
              </div>
              <div className="text-3xl font-bold text-cyan-500 flex items-baseline gap-1">
                {data?.selected_date_weather?.wind_speed.toFixed(1)}
                <span className="text-sm text-white/40">km/h</span>
              </div>
              <div className="h-1 bg-white/5 rounded-full mt-3 overflow-hidden">
                <div className="h-full w-1/2 bg-gradient-to-r from-cyan-500 to-pink-500 rounded-full" />
              </div>
            </div>
          </div>
        </div>

        {/* Risk Panel */}
        <aside className="w-[400px] bg-black/50 backdrop-blur-lg border-l border-white/10">
          <DiseaseRiskPanel
            location="Ethiopia"
            timestamp={Math.floor(date.getTime() / 1000)}
            predictions={data?.predictions}
            isLoading={isLoading}
            error={error as Error | null}
          />
        </aside>
      </main>
    </div>
  );
}

export default function Home() {
  return (
    <QueryClientProvider client={queryClient}>
      <HomePage />
    </QueryClientProvider>
  );
}
