import { Suspense } from "react";
import { ChatWorkspace } from "@/components/chat-workspace";

export default async function ThreadPage({ params }: PageProps<"/studio/[threadId]">) {
  const { threadId } = await params;
  return (
    <Suspense>
      <ChatWorkspace threadId={threadId} />
    </Suspense>
  );
}
