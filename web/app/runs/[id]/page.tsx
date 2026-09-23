import { RunLive } from "@/components/run-live";

export default async function RunPage({ params }: PageProps<"/runs/[id]">) {
  const { id } = await params;
  return <RunLive id={id} />;
}
