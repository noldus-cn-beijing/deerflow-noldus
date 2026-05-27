"use client";

import { MessageSquareHeart } from "lucide-react";
import { useState } from "react";

import { fetch as csrfFetch } from "@/core/api/fetcher";

type Category = "bug" | "enhancement" | "experience" | "other";

const CATEGORY_LABELS: Record<Category, string> = {
  bug: "问题反馈（Bug）",
  enhancement: "功能建议",
  experience: "使用体验",
  other: "其他",
};

interface FormData {
  title: string;
  category: Category;
  description: string;
  name: string;
  contact: string;
}

export default function FeedbackPage() {
  const [form, setForm] = useState<FormData>({
    title: "",
    category: "bug",
    description: "",
    name: "",
    contact: "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [toast, setToast] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setToast(null);

    const payload: Record<string, string | null> = {
      title: form.title.trim(),
      category: form.category,
      description: form.description.trim(),
      name: form.name.trim() || null,
      contact: form.contact.trim() || null,
    };

    try {
      const res = await csrfFetch("/api/feedback-issue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setToast({ type: "success", message: "反馈提交成功！感谢您的宝贵意见。" });
        setForm({
          title: "",
          category: "bug",
          description: "",
          name: "",
          contact: "",
        });
      } else {
        throw new Error(data.detail ?? "提交失败");
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "未知错误";
      setToast({
        type: "error",
        message: `提交失败: ${message}。请重试或联系管理员。`,
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-0 flex-1 items-center justify-center p-6">
      <div className="w-full max-w-lg rounded-xl border bg-card p-8 shadow-modal">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-3 flex size-12 items-center justify-center rounded-full bg-brand/10">
            <MessageSquareHeart className="size-6 text-brand" />
          </div>
          <h1 className="text-xl font-bold text-foreground">EthoInsight 反馈</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            您的意见帮助我们改进产品
          </p>
        </div>

        {toast && (
          <div
            className={`mb-5 rounded-lg px-4 py-3 text-center text-sm ${
              toast.type === "success"
                ? "border border-green-200 bg-green-50 text-green-800"
                : "border border-red-200 bg-red-50 text-red-800"
            }`}
          >
            {toast.message}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="mb-1.5 block text-sm font-semibold text-foreground">
              标题 <span className="text-destructive">*</span>
            </label>
            <input
              type="text"
              maxLength={200}
              required
              value={form.title}
              onChange={(e) => setForm({ ...form, title: e.target.value })}
              placeholder="一句话描述您遇到的问题或建议"
              className="w-full rounded-lg border border-input bg-muted/50 px-3 py-2.5 text-sm text-foreground outline-none transition-colors focus:border-ring focus:bg-background focus:ring-2 focus:ring-ring/20"
            />
          </div>

          <div>
            <label className="mb-1.5 block text-sm font-semibold text-foreground">
              反馈类型 <span className="text-destructive">*</span>
            </label>
            <select
              required
              value={form.category}
              onChange={(e) =>
                setForm({ ...form, category: e.target.value as Category })
              }
              className="w-full rounded-lg border border-input bg-muted/50 px-3 py-2.5 text-sm text-foreground outline-none transition-colors focus:border-ring focus:bg-background focus:ring-2 focus:ring-ring/20"
            >
              {(Object.entries(CATEGORY_LABELS) as [Category, string][]).map(
                ([key, label]) => (
                  <option key={key} value={key}>
                    {label}
                  </option>
                ),
              )}
            </select>
          </div>

          <div>
            <label className="mb-1.5 block text-sm font-semibold text-foreground">
              详细描述 <span className="text-destructive">*</span>
            </label>
            <textarea
              maxLength={5000}
              required
              value={form.description}
              onChange={(e) =>
                setForm({ ...form, description: e.target.value })
              }
              placeholder="请详细描述：您做了什么操作、期望什么结果、实际发生了什么"
              className="min-h-[140px] w-full resize-y rounded-lg border border-input bg-muted/50 px-3 py-2.5 text-sm text-foreground outline-none transition-colors focus:border-ring focus:bg-background focus:ring-2 focus:ring-ring/20"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-1.5 block text-sm font-semibold text-foreground">
                您的姓名
              </label>
              <input
                type="text"
                maxLength={100}
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="方便我们跟进沟通"
                className="w-full rounded-lg border border-input bg-muted/50 px-3 py-2.5 text-sm text-foreground outline-none transition-colors focus:border-ring focus:bg-background focus:ring-2 focus:ring-ring/20"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-sm font-semibold text-foreground">
                联系方式
              </label>
              <input
                type="text"
                maxLength={200}
                value={form.contact}
                onChange={(e) => setForm({ ...form, contact: e.target.value })}
                placeholder="邮箱或企业微信"
                className="w-full rounded-lg border border-input bg-muted/50 px-3 py-2.5 text-sm text-foreground outline-none transition-colors focus:border-ring focus:bg-background focus:ring-2 focus:ring-ring/20"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-lg bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? "提交中..." : "提交反馈"}
          </button>
        </form>
      </div>
    </div>
  );
}
