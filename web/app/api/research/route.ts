import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const response = await fetch("http://localhost:8000/research", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const responseData = await response.json();

    if (!response.ok) {
      // Handle FastAPI error response
      const errorMessage =
        responseData.detail ||
        (typeof responseData === "object"
          ? JSON.stringify(responseData)
          : "Failed to fetch research data");
      throw new Error(errorMessage);
    }

    return NextResponse.json(responseData);
  } catch (error) {
    console.error("Research API error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "An error occurred" },
      { status: 500 }
    );
  }
}
